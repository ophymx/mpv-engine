//! Exported-frame render backend (`export` feature, macOS + Linux): the
//! engine owns a hidden GL context on a dedicated render thread — CGL
//! on macOS, surfaceless EGL over a DRM render node on Linux — mpv
//! renders into exportable framebuffers there (IOSurface-backed /
//! DMA-BUF-backed), and the shell receives zero-copy [`ExportedFrame`]
//! handles it imports into Metal / Vulkan (or wgpu) directly — no pixel
//! ever crosses the CPU, and no GL leaks into the shell.
//!
//! Why this shape: libmpv's render API speaks only OpenGL and software,
//! and mpv never inspects what memory backs the FBO it is handed — so
//! the consumer-side fix for "no Metal/Vulkan render API" is to make the
//! FBO's color attachment *born exportable* (an IOSurface / a DMA-BUF)
//! and hand the handle across. The GL involvement is confined to this
//! module's thread; the shell's compositor never touches it.
//!
//! Pool discipline: a small ring of buffers (default 3) rotates through
//! free → rendering → published → in-use(shell) → free. mpv renders into
//! a free buffer, the platform's publish barrier makes it coherent for
//! other APIs (`glFlush` under IOSurface's coherency contract on macOS,
//! `glFinish` on Linux — see each platform module), and the newest
//! published frame replaces an unconsumed older one — the shell always
//! acquires the latest frame. While the shell holds every buffer,
//! frames are dropped, not queued.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(all(feature = "wgpu", target_os = "linux"))]
mod wgpu_linux;
#[cfg(all(feature = "wgpu", target_os = "macos"))]
mod wgpu_macos;

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
use rsmpv::Mpv;
use rsmpv::render::{OpenGlFbo, OwnedRenderContext};

use crate::error::{Error, Result};
use platform::SurfaceBuffer;

/// Attach-time configuration for
/// [`Engine::attach_exported_render`](crate::Engine::attach_exported_render).
///
/// Non-exhaustive so future knobs (pixel format, colorspace) stay
/// additive — construct with [`new`](Self::new) plus the chainable
/// setters.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ExportOptions {
    /// Initial frame width in pixels
    /// ([`Engine::set_export_size`](crate::Engine::set_export_size)
    /// changes it later).
    pub width: u32,
    /// Initial frame height in pixels.
    pub height: u32,
    /// Buffers in rotation (minimum 2, default 3): one being rendered,
    /// one published, one in the shell's hands. Raise it only if the
    /// shell legitimately holds several frames at once (e.g. frames in
    /// flight across a deep GPU pipeline).
    pub pool_size: usize,
}

impl ExportOptions {
    /// Options for `width`×`height` frames with the default pool of 3.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pool_size: 3,
        }
    }

    /// Set [`pool_size`](field@Self::pool_size) (clamped to at least 2),
    /// chainable from [`new`](Self::new).
    #[must_use]
    pub fn pool_size(mut self, pool_size: usize) -> Self {
        self.pool_size = pool_size.max(2);
        self
    }
}

fn pack_size(width: u32, height: u32) -> u64 {
    (u64::from(width) << 32) | u64::from(height)
}

fn unpack_size(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, packed as u32)
}

/// State shared by the engine's backend handle, the render thread, and
/// every outstanding [`ExportedFrame`].
pub(crate) struct ExportShared {
    state: Mutex<PoolState>,
    /// Wakes the render thread; every flag below is checked under
    /// `state`'s lock in its wait loop.
    cond: Condvar,
    shutdown: AtomicBool,
    /// An mpv update callback fired since the last service.
    update_pending: AtomicBool,
    /// Render a frame even without a new-frame update — set on resize so
    /// a paused/ended video re-renders at the new size.
    force_render: AtomicBool,
    /// A frame was `presented()`; forward `report_swap` to mpv.
    swap_pending: AtomicBool,
    /// A wanted render was dropped because no pool buffer was available
    /// (shell holding all of them, or allocation failed). Checked when a
    /// buffer comes back ([`return_buffer`] / [`retire_buffer`]), which
    /// re-arms `force_render` — so a `set_export_size` landing during
    /// pool exhaustion still re-renders once a buffer frees up, instead
    /// of leaving a paused/ended video stale at the old size forever.
    render_dropped: AtomicBool,
    /// Current target size, `(w << 32) | h`. Applied from the next
    /// rendered frame; stale-size pool buffers are retired lazily.
    target_size: AtomicU64,
    pool_size: usize,
}

#[derive(Default)]
struct PoolState {
    /// Buffers ready to render into (possibly of a stale size — the
    /// render thread retires those when it next needs a buffer).
    free: Vec<SurfaceBuffer>,
    /// The newest rendered frame the shell has not acquired yet.
    published: Option<SurfaceBuffer>,
    /// Frames currently in the shell's hands.
    in_use: usize,
    /// Buffers alive in total (`free` + `published` + `in_use`) — the
    /// pool-size cap counts all of them. Buffers in `retired` have
    /// already left the count.
    live: usize,
    /// Buffers permanently out of the pool (the Linux wgpu import
    /// consumes them — see [`retire_buffer`]), parked here for the
    /// render thread to delete their GL/EGL names on its next wake.
    /// Always empty on macOS.
    retired: Vec<SurfaceBuffer>,
}

impl ExportShared {
    fn notify(&self) {
        // Lock-then-notify so a render thread between its flag checks and
        // its wait can't miss the signal.
        let _state = self.state.lock();
        self.cond.notify_all();
    }

    pub(crate) fn set_target_size(&self, width: u32, height: u32) {
        self.target_size
            .store(pack_size(width, height), Ordering::SeqCst);
        self.force_render.store(true, Ordering::SeqCst);
        self.notify();
    }

    /// A published frame is waiting to be acquired. Used by attach to
    /// re-fire `on_update` for a frame published before the backend
    /// landed in the engine's render slot.
    pub(crate) fn has_published(&self) -> bool {
        self.state.lock().published.is_some()
    }

    /// A buffer became available again: if a wanted render was dropped
    /// for lack of one, re-arm the forced render now.
    fn rearm_dropped_render(&self) {
        if self.render_dropped.swap(false, Ordering::SeqCst) {
            self.force_render.store(true, Ordering::SeqCst);
            self.notify();
        }
    }
}

/// The engine-side handle stored in the render slot: shared state plus
/// the join handle. Dropping it (detach, or engine drop) shuts the
/// render thread down and joins it — which is exactly the "free the
/// render context with its GL context current" teardown, performed on
/// the one thread where that context is current.
pub(crate) struct ExportedRender {
    shared: Arc<ExportShared>,
    thread: Option<JoinHandle<()>>,
}

impl ExportedRender {
    /// Spawn the render thread and wait for its GL context + mpv render
    /// context to come up; `Err` means nothing was attached (the thread
    /// has already exited).
    pub(crate) fn create(
        core: Arc<Mpv>,
        options: ExportOptions,
        relay: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self> {
        let shared = Arc::new(ExportShared {
            state: Mutex::new(PoolState::default()),
            cond: Condvar::new(),
            shutdown: AtomicBool::new(false),
            update_pending: AtomicBool::new(false),
            force_render: AtomicBool::new(false),
            swap_pending: AtomicBool::new(false),
            render_dropped: AtomicBool::new(false),
            target_size: AtomicU64::new(pack_size(options.width, options.height)),
            pool_size: options.pool_size.max(2),
        });
        let (init_tx, init_rx) = std::sync::mpsc::channel();
        let thread_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("mpv-export-render".into())
            .spawn(move || render_thread(core, thread_shared, Box::new(relay), init_tx))
            .map_err(|e| Error::ExportSetup(format!("spawning render thread: {e}")))?;
        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                shared,
                thread: Some(thread),
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(Error::ExportSetup("render thread died during setup".into()))
            }
        }
    }

    pub(crate) fn shared(&self) -> &Arc<ExportShared> {
        &self.shared
    }

    /// Whether the calling thread *is* this backend's render thread —
    /// i.e. we are inside `on_update` (or code it called). Drives the
    /// two teardown paths below and
    /// [`Engine::detach_render`](crate::Engine::detach_render)'s lock
    /// choice.
    pub(crate) fn is_render_thread(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|t| t.thread().id() == std::thread::current().id())
    }
}

impl Drop for ExportedRender {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.notify();
        if let Some(thread) = self.thread.take() {
            if thread.thread().id() == std::thread::current().id() {
                // Dropped from inside the render thread's own callback (a
                // shell detaching — or dropping its last Engine — from
                // `on_update`): joining would self-deadlock (EDEADLK, a
                // panic inside Drop). Detach the handle instead; the
                // shutdown flag is already set, so the thread runs its
                // normal teardown and exits as soon as the callback
                // returns. The GL context is freed on the thread it is
                // current on either way.
                return;
            }
            let _ = thread.join();
        }
    }
}

/// The render thread: owns the hidden GL context (CGL / EGL) and the
/// mpv render context for their whole lives, so GL-currency is a
/// per-thread invariant here rather than a caller obligation — this is
/// what lets the public attach be a safe fn.
fn render_thread(
    core: Arc<Mpv>,
    shared: Arc<ExportShared>,
    relay: Box<dyn Fn() + Send + Sync>,
    init_tx: std::sync::mpsc::Sender<Result<()>>,
) {
    let gl = match platform::GlContext::new() {
        Ok(gl) => gl,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };
    if let Err(e) = gl.make_current() {
        let _ = init_tx.send(Err(e));
        return;
    }
    // Advanced control on: this thread services `update()` promptly after
    // every callback by construction, and mpv gets direct rendering.
    let proc_address: Box<dyn FnMut(&str) -> *mut c_void + Send + 'static> =
        Box::new(platform::gl_proc_address);
    // SAFETY: the GL context above is current on this thread now and for
    // every later call — context and renderer live and die on this one
    // thread, with the renderer dropped first (declaration order below is
    // irrelevant; both drops happen before `render_thread` returns, ctx
    // explicitly before `gl`).
    let mut ctx = match unsafe { OwnedRenderContext::new_opengl(core, true, proc_address) } {
        Ok(ctx) => ctx,
        Err(e) => {
            let _ = init_tx.send(Err(e.into()));
            return;
        }
    };
    {
        let signal = Arc::clone(&shared);
        // Fires on mpv's internal threads (and once synchronously right
        // here): flag-and-wake only, per the render API's rules.
        ctx.set_update_callback(move || {
            signal.update_pending.store(true, Ordering::SeqCst);
            signal.notify();
        });
    }
    let _ = init_tx.send(Ok(()));

    loop {
        {
            let mut state = shared.state.lock();
            while !shared.shutdown.load(Ordering::SeqCst)
                && !shared.update_pending.load(Ordering::SeqCst)
                && !shared.force_render.load(Ordering::SeqCst)
                && !shared.swap_pending.load(Ordering::SeqCst)
                && state.retired.is_empty()
            {
                shared.cond.wait(&mut state);
            }
        }
        drain_retired(&shared, &gl);
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        if shared.swap_pending.swap(false, Ordering::SeqCst) {
            ctx.report_swap();
        }
        let force = shared.force_render.swap(false, Ordering::SeqCst);
        let updated = shared.update_pending.swap(false, Ordering::SeqCst);
        // `update()` must run for every callback (advanced control); it
        // also gates rendering to real new-frame updates.
        let wants_frame = updated && ctx.update();
        if wants_frame || force {
            render_one(&mut ctx, &gl, &shared, &relay);
        }
    }

    // Teardown, still on this thread with the GL context current: pooled
    // buffers' GL names first, then the render context (the crate's
    // free-with-context-current contract), then the context itself.
    // Frames still in the shell's hands keep their backing memory alive
    // on their own (retained IOSurface / dmabuf fd); their GL names die
    // with the context.
    {
        let mut state = shared.state.lock();
        for mut buffer in state.free.drain(..) {
            buffer.delete_gl(&gl);
        }
        for mut buffer in state.retired.drain(..) {
            buffer.delete_gl(&gl);
        }
        if let Some(mut buffer) = state.published.take() {
            buffer.delete_gl(&gl);
        }
    }
    drop(ctx);
    drop(gl);
}

/// Delete the GL/EGL names of buffers retired out of the pool by other
/// threads (see [`retire_buffer`]) and drop them. Render thread only,
/// GL context current; dropping also closes the buffer's fd — safe,
/// because a retired buffer's memory is pinned by the importer's own
/// reference.
fn drain_retired(shared: &ExportShared, gl: &platform::GlContext) {
    let retired = std::mem::take(&mut shared.state.lock().retired);
    for mut buffer in retired {
        buffer.delete_gl(gl);
    }
}

/// Render the current frame into a pool buffer and publish it. Skips
/// (dropping the frame) when the shell is holding every buffer.
fn render_one(
    ctx: &mut OwnedRenderContext,
    gl: &platform::GlContext,
    shared: &Arc<ExportShared>,
    relay: &dyn Fn(),
) {
    let (width, height) = unpack_size(shared.target_size.load(Ordering::SeqCst));
    if width == 0 || height == 0 {
        return;
    }
    let buffer = {
        let mut state = shared.state.lock();
        // Retire free buffers from before a resize (GL deletion is legal
        // here: this is the render thread, context current).
        let mut kept = Vec::with_capacity(state.free.len());
        for mut buffer in std::mem::take(&mut state.free) {
            if buffer.size() == (width, height) {
                kept.push(buffer);
            } else {
                buffer.delete_gl(gl);
                state.live -= 1;
            }
        }
        state.free = kept;
        if let Some(buffer) = state.free.pop() {
            Some(buffer)
        } else {
            // Steal an unconsumed published frame before growing: the
            // shell skipped it, and newest-wins is the display policy.
            if let Some(mut stale) = state.published.take_if(|b| b.size() != (width, height)) {
                stale.delete_gl(gl);
                state.live -= 1;
            }
            if let Some(buffer) = state.published.take() {
                Some(buffer)
            } else if state.live < shared.pool_size {
                match SurfaceBuffer::new(gl, width, height) {
                    Ok(buffer) => {
                        state.live += 1;
                        Some(buffer)
                    }
                    Err(e) => {
                        tracing::warn!("export: buffer allocation failed: {e}");
                        shared.render_dropped.store(true, Ordering::SeqCst);
                        None
                    }
                }
            } else {
                tracing::debug!(
                    "export: shell holds all {} buffers; dropping frame",
                    shared.pool_size
                );
                // The consumed update/force flags would otherwise lose
                // this render for good on a paused/ended video; a buffer
                // return re-arms it (see `render_dropped`).
                shared.render_dropped.store(true, Ordering::SeqCst);
                None
            }
        }
    };
    let Some(buffer) = buffer else { return };
    let fbo = OpenGlFbo {
        fbo: buffer.fbo() as i32,
        width: width as i32,
        height: height as i32,
        internal_format: platform::FBO_INTERNAL_FORMAT,
    };
    // No flip: mpv's unflipped FBO output already puts row 0 at the top
    // of the image (pinned by the orientation integration tests on both
    // platforms), which is the layout Metal/CoreVideo/Vulkan consumers
    // read; flip_y is for GL-convention targets. Block-for-target-time
    // is mpv's own pacing, harmless on this dedicated thread.
    if let Err(e) = ctx.render_opengl(fbo, false, true) {
        tracing::warn!("export render failed: {e}");
        shared.state.lock().free.push(buffer);
        return;
    }
    gl.publish_barrier();
    {
        let mut state = shared.state.lock();
        if let Some(previous) = state.published.replace(buffer) {
            state.free.push(previous);
        }
    }
    relay();
}

/// One rendered video frame, wrapping the exportable buffer the
/// engine's hidden GL context rendered into — a retained IOSurface on
/// macOS, a DMA-BUF on Linux. Acquired with
/// [`Engine::acquire_frame`](crate::Engine::acquire_frame), imported
/// into the shell's GPU API via the platform handle (`io_surface` on
/// macOS, `dma_buf_fd` on Linux).
///
/// Dropping the frame returns its buffer to the render pool, after which
/// **mpv will render future frames into the same memory** — hold the
/// frame until the GPU work sampling it has completed (e.g. until the
/// command buffer's completion handler / fence), not merely until it
/// was encoded. Importing the handle into Metal or Vulkan takes its own
/// reference on the memory, but that only keeps the memory alive; it
/// does not stop the pool reusing it for pixels.
///
/// With the `wgpu` feature, `into_wgpu_texture` removes that whole
/// obligation: wgpu's own completion tracking keeps the memory backing
/// the imported texture out of mpv's hands until wgpu has finished with
/// it, GPU work included (returned to the pool on macOS, retired from
/// it on Linux — invisible either way).
pub struct ExportedFrame {
    buffer: Option<SurfaceBuffer>,
    shared: Arc<ExportShared>,
}

impl ExportedFrame {
    pub(crate) fn take_published(shared: Arc<ExportShared>) -> Option<Self> {
        let mut state = shared.state.lock();
        let buffer = state.published.take()?;
        state.in_use += 1;
        drop(state);
        Some(Self {
            buffer: Some(buffer),
            shared,
        })
    }

    fn buffer(&self) -> &SurfaceBuffer {
        self.buffer
            .as_ref()
            .expect("buffer present until drop/presented")
    }

    /// The frame's retained `IOSurfaceRef`, valid while this frame is
    /// alive (see the type docs for the reuse hazard after drop). Pass it
    /// to `MTLDevice newTextureWithDescriptor:iosurface:plane:` (pixel
    /// format `bgra8Unorm`, plane 0) or your interop layer's equivalent.
    #[cfg(target_os = "macos")]
    pub fn io_surface(&self) -> *mut c_void {
        self.buffer().io_surface()
    }

    /// The frame's DMA-BUF fd, valid while this frame is alive (see the
    /// type docs for the reuse hazard after drop). A single-plane
    /// [`fourcc`](Self::fourcc) buffer with [`modifier`](Self::modifier)
    /// `DRM_FORMAT_MOD_LINEAR`, [`stride`](Self::stride) bytes per row,
    /// plane offset 0.
    ///
    /// Import it into Vulkan via `VK_EXT_external_memory_dma_buf`
    /// (`VK_FORMAT_B8G8R8A8_UNORM`, `VK_IMAGE_TILING_LINEAR`), into EGL
    /// via `EGL_EXT_image_dma_buf_import`, or across a compositor
    /// protocol — anything that speaks dmabuf. APIs that take fd
    /// *ownership* (Vulkan import does) must be handed a duplicate
    /// (`try_clone_to_owned`), never this fd.
    #[cfg(target_os = "linux")]
    pub fn dma_buf_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.buffer().dma_buf_fd()
    }

    /// Bytes per row of the DMA-BUF's single plane (≥ `width * 4`).
    #[cfg(target_os = "linux")]
    pub fn stride(&self) -> u32 {
        self.buffer().stride()
    }

    /// The buffer's DRM fourcc: `DRM_FORMAT_ARGB8888` (`'AR24'`) — or
    /// `DRM_FORMAT_XRGB8888` (`'XR24'`) on drivers that couldn't render
    /// to the former, in which case the alpha byte is meaningless. Both
    /// are little-endian B,G,R,A/X bytes: `VK_FORMAT_B8G8R8A8_UNORM` /
    /// wgpu `Bgra8Unorm`.
    #[cfg(target_os = "linux")]
    pub fn fourcc(&self) -> u32 {
        self.buffer().fourcc()
    }

    /// The buffer's DRM format modifier — always
    /// `DRM_FORMAT_MOD_LINEAR` (0) with this backend, exposed so import
    /// code can pass it through instead of hardcoding.
    #[cfg(target_os = "linux")]
    pub fn modifier(&self) -> u64 {
        platform::MODIFIER_LINEAR
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.buffer().size().0
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.buffer().size().1
    }

    /// Copy the pixels out as tightly packed BGRA8 rows, row 0 at the
    /// top. This is a CPU readback — for screenshots, thumbnails, and
    /// tests, not the per-frame display path (that's what the zero-copy
    /// platform handle avoids).
    pub fn copy_pixels(&self) -> Vec<u8> {
        self.buffer().copy_pixels()
    }

    /// Consume the frame, telling mpv it was actually displayed
    /// (`mpv_render_context_report_swap` — feeds its frame-timing
    /// statistics). Optional: plain `drop` returns the buffer without
    /// the report.
    pub fn presented(self) {
        self.shared.swap_pending.store(true, Ordering::SeqCst);
        self.shared.notify();
        // Drop returns the buffer to the pool.
    }
}

impl Drop for ExportedFrame {
    fn drop(&mut self) {
        // `buffer` is `None` when ownership moved on — into the wgpu
        // drop callback (`into_wgpu_texture`), which does this return
        // itself once wgpu is finished with the texture.
        if let Some(buffer) = self.buffer.take() {
            return_buffer(&self.shared, buffer);
        }
    }
}

/// Return an in-use buffer to the pool: the one bookkeeping path shared
/// by [`ExportedFrame`]'s drop and (on macOS) the wgpu import's drop
/// callback. After teardown the buffer just idles in `free` until the
/// shared state drops with it (GL names died with the context; the
/// backing memory is released by `SurfaceBuffer`'s own drop).
fn return_buffer(shared: &ExportShared, buffer: SurfaceBuffer) {
    {
        let mut state = shared.state.lock();
        state.in_use -= 1;
        state.free.push(buffer);
    }
    shared.rearm_dropped_render();
}

/// Permanently retire an in-use buffer from the pool — the Linux wgpu
/// import's counterpart to [`return_buffer`]: the dmabuf's memory now
/// backs a wgpu texture that wgpu releases on its own completion
/// schedule, so mpv must never render into that memory again (the same
/// aliasing hazard the macOS drop callback prevents, solved by
/// consumption instead of reuse). Shrinking `live` lets the render
/// thread allocate a replacement; the buffer parks in `retired` until
/// the render thread deletes its GL/EGL names and closes our fd — the
/// importer holds its own duplicate.
#[cfg(all(feature = "wgpu", target_os = "linux"))]
fn retire_buffer(shared: &ExportShared, buffer: SurfaceBuffer) {
    {
        let mut state = shared.state.lock();
        state.in_use -= 1;
        state.live -= 1;
        state.retired.push(buffer);
    }
    // `live` shrank, so capacity exists again — a dropped render can
    // re-run into a fresh allocation. The notify inside doubles as the
    // retired-queue wake.
    shared.rearm_dropped_render();
    shared.notify();
}

impl std::fmt::Debug for ExportedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedFrame")
            .field("width", &self.width())
            .field("height", &self.height())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_packing_round_trips() {
        assert_eq!(unpack_size(pack_size(3840, 2160)), (3840, 2160));
        assert_eq!(unpack_size(pack_size(0, 0)), (0, 0));
        assert_eq!(unpack_size(pack_size(u32::MAX, 1)), (u32::MAX, 1));
    }

    #[test]
    fn pool_size_clamps_to_two() {
        assert_eq!(ExportOptions::new(64, 64).pool_size(0).pool_size, 2);
        assert_eq!(ExportOptions::new(64, 64).pool_size(5).pool_size, 5);
    }
}
