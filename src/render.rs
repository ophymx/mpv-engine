//! Render seam — OpenGL and software backends over rsmpv's
//! [`OwnedRenderContext`].
//!
//! Since rsmpv 0.2 the safe render context comes in an owned flavor that
//! co-owns the core through `Arc<Mpv>` and is `Send` — the two upstream
//! changes this module's previous raw-`sys` incarnation was waiting for.
//! Free-before-terminate ordering is now structural (the context's `Arc`
//! keeps the core alive until the context drops), and what remains here is
//! the backend-specific param plumbing: fbo/flip wiring for GL, dimension
//! guards and the `rgb0` alpha quirk for software.
//!
//! Threading contract (unchanged): the update callback fires on **mpv's
//! render thread**, and also *synchronously during registration* —
//! consumers must be re-entrant-safe at attach time. For the GL backend,
//! the target GL context must be current for creation, every render, and
//! teardown — `mpv_render_context_free` tears down GL objects, and freeing
//! without the right context current leaks them into whatever context *is*
//! current (in GTK that manifested as whole-window rendering artifacts
//! after the player page was popped). rsmpv encodes that per-call rule as
//! an `unsafe` GL constructor; this crate forwards the obligation through
//! [`Engine::attach_gl_render`](crate::Engine::attach_gl_render) rather
//! than hiding it behind a safe fn that could still hit undefined
//! behavior.

use std::ffi::c_void;
use std::sync::Arc;

use rsmpv::Mpv;
use rsmpv::render::{OpenGlFbo, OwnedRenderContext, SwPixelFormat};

use crate::error::Result;

/// Resolves GL symbols for mpv. Called during context creation (and
/// possibly later render calls), so it must stay alive for the context's
/// lifetime — rsmpv keeps it boxed inside the context.
pub type ProcAddressFn = Box<dyn FnMut(&str) -> *mut c_void + Send + 'static>;

/// Attach-time knobs for the OpenGL backend
/// ([`Engine::attach_gl_render`](crate::Engine::attach_gl_render)). The
/// default is mpv's stock behavior — right for a toolkit paint handler
/// (GTK GLArea); shells whose render loop must not stall override per
/// field. Attach-time on purpose: frame pacing is a property of the
/// shell's render loop, not of any single frame.
///
/// Non-exhaustive so future knobs stay additive — which also forbids
/// struct expressions outside this crate (E0639, functional record
/// update included), so construct through the chainable setters:
/// `GlRenderOptions::default().block_for_target_time(false)`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GlRenderOptions {
    /// Block inside [`render_gl`](crate::Engine::render_gl) until the
    /// frame's target display time — mpv's default, and the right pacing
    /// when the toolkit's frame clock drives drawing. Set `false` for
    /// render loops that must not stall (e.g. preparing inside a
    /// compositor-thread pass, where blocking would hold up the whole
    /// scene submit): mpv then returns immediately and frame pacing is
    /// yours — do your own timing, or set the `video-timing-offset`
    /// property to `0` (mpv's documented alternative).
    pub block_for_target_time: bool,
    /// mpv's `MPV_RENDER_PARAM_ADVANCED_CONTROL`: enables direct
    /// rendering and GPU screenshots, but obligates the shell to follow
    /// the render API threading rules strictly and to call
    /// [`render_update`](crate::Engine::render_update) promptly after
    /// **every** update callback (optional when this is off).
    pub advanced_control: bool,
}

impl Default for GlRenderOptions {
    fn default() -> Self {
        Self {
            block_for_target_time: true,
            advanced_control: false,
        }
    }
}

impl GlRenderOptions {
    /// Set [`block_for_target_time`](field@Self::block_for_target_time),
    /// chainable from [`default()`](Default::default).
    #[must_use]
    pub fn block_for_target_time(mut self, block: bool) -> Self {
        self.block_for_target_time = block;
        self
    }

    /// Set [`advanced_control`](field@Self::advanced_control), chainable
    /// from [`default()`](Default::default).
    #[must_use]
    pub fn advanced_control(mut self, advanced: bool) -> Self {
        self.advanced_control = advanced;
        self
    }
}

/// Which render backend is attached — what
/// [`Engine::attached_render`](crate::Engine::attached_render) reports.
///
/// Non-exhaustive: a future backend (e.g. Vulkan, if mpv ever exposes it
/// through the render API) is an additive variant, not a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderKind {
    /// The OpenGL backend
    /// ([`Engine::attach_gl_render`](crate::Engine::attach_gl_render)).
    OpenGl,
    /// The software backend
    /// ([`Engine::attach_sw_render`](crate::Engine::attach_sw_render)).
    Software,
    /// The exported-frame backend (`export` feature, macOS + Linux +
    /// Windows — `Engine::attach_exported_render`): the engine renders on
    /// its own hidden GL context and hands the shell zero-copy exportable
    /// frames (IOSurface-backed / DMA-BUF-backed / shared-D3D11-texture-
    /// backed). The variant exists on every platform so cross-platform
    /// shells can match on it unconditionally; only the feature produces
    /// it.
    Exported,
}

/// The one attached render backend. Backends share the engine's single
/// slot so `AlreadyAttached` and `detach_render` behave uniformly —
/// mpv allows one render context per handle regardless of type.
pub(crate) enum RenderBackend {
    Gl(GlRender),
    Sw(SwRender),
    #[cfg(export_backend)]
    Exported(crate::export::ExportedRender),
}

impl RenderBackend {
    /// Process pending render work (`mpv_render_context_update`);
    /// `true` when a new frame should be drawn. For the exported backend
    /// this is a no-op returning `false`: its render context lives on the
    /// engine's own render thread, which services updates itself.
    pub(crate) fn update(&mut self) -> bool {
        match self {
            RenderBackend::Gl(r) => r.ctx.update(),
            RenderBackend::Sw(r) => r.0.update(),
            #[cfg(export_backend)]
            RenderBackend::Exported(_) => false,
        }
    }

    pub(crate) fn kind(&self) -> RenderKind {
        match self {
            RenderBackend::Gl(_) => RenderKind::OpenGl,
            RenderBackend::Sw(_) => RenderKind::Software,
            #[cfg(export_backend)]
            RenderBackend::Exported(_) => RenderKind::Exported,
        }
    }

    /// Whether the calling thread is this backend's own render thread —
    /// true only for the exported backend when called from inside its
    /// `on_update`. Decides `detach_render`'s locking (see there): a
    /// self-detach must not block on the attach lock, which a concurrent
    /// detach may hold while joining this very thread.
    pub(crate) fn on_own_render_thread(&self) -> bool {
        match self {
            #[cfg(export_backend)]
            RenderBackend::Exported(r) => r.is_render_thread(),
            _ => false,
        }
    }
}

pub(crate) struct GlRender {
    ctx: OwnedRenderContext,
    block_for_target_time: bool,
}

impl GlRender {
    /// Create an OpenGL render context co-owning `core` and register
    /// `on_update`.
    ///
    /// `on_update` fires once synchronously here (mpv's documented
    /// behavior) and afterwards from the render thread.
    ///
    /// # Safety
    /// Forwards rsmpv's `new_opengl` contract: the target GL context must
    /// be current on the calling thread now, on every later
    /// [`render`](Self::render) or update, and when this value drops.
    pub(crate) unsafe fn create(
        core: Arc<Mpv>,
        get_proc_address: ProcAddressFn,
        options: GlRenderOptions,
        on_update: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self> {
        // SAFETY: GL-currency contract forwarded to the caller.
        let mut ctx = unsafe {
            OwnedRenderContext::new_opengl(core, options.advanced_control, get_proc_address)?
        };
        ctx.set_update_callback(on_update);
        Ok(Self {
            ctx,
            block_for_target_time: options.block_for_target_time,
        })
    }

    /// Draw the current frame into `fbo` (`0` = default framebuffer).
    /// `flip_y` handles targets with a flipped origin (e.g. GTK's GLArea).
    /// Whether this blocks until the frame's target time was fixed at
    /// attach ([`GlRenderOptions::block_for_target_time`]).
    pub(crate) fn render(&mut self, fbo: i32, w: i32, h: i32, flip_y: bool) -> Result<()> {
        let fbo = OpenGlFbo {
            fbo,
            width: w,
            height: h,
            internal_format: 0,
        };
        self.ctx
            .render_opengl(fbo, flip_y, self.block_for_target_time)?;
        Ok(())
    }
}

/// Software rendering: mpv draws the frame into a caller-provided RGBA
/// buffer. No GL anywhere — no context-current requirements for rendering
/// *or* teardown, so unlike [`GlRender`] this backend is fully safe and
/// drops from any thread.
pub(crate) struct SwRender(OwnedRenderContext);

impl SwRender {
    /// Create a software render context co-owning `core` and register
    /// `on_update` (same contract as the GL backend: fires once
    /// synchronously here, afterwards from mpv's render thread).
    pub(crate) fn create(
        core: Arc<Mpv>,
        on_update: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self> {
        let mut ctx = OwnedRenderContext::new_software(core)?;
        ctx.set_update_callback(on_update);
        Ok(Self(ctx))
    }

    /// Render the current frame as RGBA8 into `buf`, resizing it to
    /// `w * h * 4`.
    pub(crate) fn render(&mut self, w: i32, h: i32, buf: &mut Vec<u8>) -> Result<()> {
        // Nothing to draw for empty or negative dimensions — rsmpv would
        // reject them as InvalidParameter, but a no-op mirrors this
        // crate's render-before-attach philosophy. The overflow check
        // keeps the multiply sound on 32-bit targets.
        let (Ok(uw), Ok(uh)) = (usize::try_from(w), usize::try_from(h)) else {
            return Ok(());
        };
        let Some(len) = uw.checked_mul(uh).and_then(|p| p.checked_mul(4)) else {
            return Ok(());
        };
        if len == 0 {
            return Ok(());
        }
        buf.resize(len, 0);
        self.0
            .render_software(w, h, SwPixelFormat::Rgb0, uw * 4, buf)?;
        // "rgb0" leaves the fourth byte of each pixel undefined; a
        // consumer treating the buffer as RGBA reads it as alpha and
        // gets garbage transparency. Force opaque here — the quirk
        // belongs to the seam, not to every consumer.
        for px in buf.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
        Ok(())
    }
}
