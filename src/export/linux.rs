//! Raw Linux plumbing for the exported-frame backend: a hidden EGL
//! context on a DRM render node (GBM platform, surfaceless — no
//! compositor or display server involved), GL framebuffers whose color
//! storage is a GBM-allocated DMA-BUF, and the EGL symbol loader mpv
//! renders through. libEGL and libgbm are linked directly — no Rust
//! dependency enters the tree for this.
//!
//! Thread discipline: everything that touches GL or EGL images —
//! context creation and currency, [`SurfaceBuffer::new`],
//! [`SurfaceBuffer::delete_gl`] — runs only on the engine's export
//! render thread (the module's only GL caller), so GL's thread rules
//! are upheld by construction rather than by types. The DMA-BUF side is
//! a plain file descriptor — kernel-refcounted and usable from any
//! thread or process, which is exactly why it is the export currency
//! (the IOSurface analog).
//!
//! Buffer lifetime: each pool buffer allocates a GBM bo, exports its
//! dmabuf fd, imports the fd back as an EGLImage bound to a GL texture,
//! and then destroys the bo immediately — the fd and the image each
//! hold their own kernel reference to the memory, so the bo handle is
//! pure scaffolding. After that the buffer owns only the fd (closed by
//! `Drop`, any thread) plus GL/EGL names only the render thread ever
//! dereferences — the same split as macOS's retained IOSurface vs. GL
//! names.

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use super::{ExportBackend, ExportBuffer};
use crate::error::{Error, Result};

// ---- GBM (libgbm) ----

#[repr(C)]
struct GbmDevice {
    _opaque: [u8; 0],
}
#[repr(C)]
struct GbmBo {
    _opaque: [u8; 0],
}

/// `GBM_BO_USE_RENDERING`.
const GBM_USE_RENDERING: u32 = 1 << 2;
/// `GBM_BO_USE_LINEAR` — see [`create_bo`] for why every buffer is
/// linear.
const GBM_USE_LINEAR: u32 = 1 << 4;

#[link(name = "gbm")]
unsafe extern "C" {
    fn gbm_create_device(fd: c_int) -> *mut GbmDevice;
    fn gbm_device_destroy(gbm: *mut GbmDevice);
    fn gbm_bo_create(
        gbm: *mut GbmDevice,
        width: u32,
        height: u32,
        format: u32,
        flags: u32,
    ) -> *mut GbmBo;
    fn gbm_bo_destroy(bo: *mut GbmBo);
    fn gbm_bo_get_fd(bo: *mut GbmBo) -> c_int;
    fn gbm_bo_get_stride(bo: *mut GbmBo) -> u32;
}

// ---- EGL (libEGL) ----

type EGLDisplay = *mut c_void;
type EGLConfig = *mut c_void;
type EGLContext = *mut c_void;
type EGLSurface = *mut c_void;
type EGLImage = *mut c_void;
type EGLBoolean = u32;
type EGLint = i32;
type EGLenum = u32;
type EGLAttrib = isize;

/// `EGL_PLATFORM_GBM_KHR`.
const EGL_PLATFORM_GBM: EGLenum = 0x31D7;
const EGL_OPENGL_API: EGLenum = 0x30A2;
const EGL_OPENGL_ES_API: EGLenum = 0x30A0;
const EGL_OPENGL_BIT: EGLint = 0x0008;
const EGL_OPENGL_ES2_BIT: EGLint = 0x0004;
const EGL_SURFACE_TYPE: EGLint = 0x3033;
const EGL_RENDERABLE_TYPE: EGLint = 0x3040;
const EGL_CONTEXT_CLIENT_VERSION: EGLint = 0x3098;
const EGL_EXTENSIONS: EGLint = 0x3055;
const EGL_NONE: EGLint = 0x3038;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
/// `EGL_EXT_image_dma_buf_import` attribute tokens.
const EGL_LINUX_DMA_BUF: EGLenum = 0x3270;
const EGL_LINUX_DRM_FOURCC: EGLint = 0x3271;
const EGL_DMA_BUF_PLANE0_FD: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH: EGLint = 0x3274;

const EGL_NO_DISPLAY: EGLDisplay = std::ptr::null_mut();
const EGL_NO_CONTEXT: EGLContext = std::ptr::null_mut();
const EGL_NO_SURFACE: EGLSurface = std::ptr::null_mut();
/// `EGL_NO_CONFIG_KHR` (`EGL_KHR_no_config_context`).
const EGL_NO_CONFIG: EGLConfig = std::ptr::null_mut();

#[link(name = "EGL")]
unsafe extern "C" {
    fn eglGetPlatformDisplay(
        platform: EGLenum,
        native_display: *mut c_void,
        attrib_list: *const EGLAttrib,
    ) -> EGLDisplay;
    fn eglInitialize(dpy: EGLDisplay, major: *mut EGLint, minor: *mut EGLint) -> EGLBoolean;
    fn eglTerminate(dpy: EGLDisplay) -> EGLBoolean;
    fn eglBindAPI(api: EGLenum) -> EGLBoolean;
    fn eglChooseConfig(
        dpy: EGLDisplay,
        attribs: *const EGLint,
        configs: *mut EGLConfig,
        config_size: EGLint,
        num_config: *mut EGLint,
    ) -> EGLBoolean;
    fn eglCreateContext(
        dpy: EGLDisplay,
        config: EGLConfig,
        share: EGLContext,
        attribs: *const EGLint,
    ) -> EGLContext;
    fn eglDestroyContext(dpy: EGLDisplay, ctx: EGLContext) -> EGLBoolean;
    fn eglMakeCurrent(
        dpy: EGLDisplay,
        draw: EGLSurface,
        read: EGLSurface,
        ctx: EGLContext,
    ) -> EGLBoolean;
    fn eglGetError() -> EGLint;
    fn eglQueryString(dpy: EGLDisplay, name: EGLint) -> *const c_char;
    fn eglGetProcAddress(name: *const c_char) -> *mut c_void;
}

type PfnEglCreateImage =
    unsafe extern "C" fn(EGLDisplay, EGLContext, EGLenum, *mut c_void, *const EGLint) -> EGLImage;
type PfnEglDestroyImage = unsafe extern "C" fn(EGLDisplay, EGLImage) -> EGLBoolean;

// ---- GL entry points (loaded, not linked: which client library backs
// them — libGL or libGLESv2 — depends on which context the ladder in
// `create_context` lands on) ----

// Texture/FBO GL enums shared with the other backends.
use super::gl_consts::{
    GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_NEAREST, GL_RGBA8,
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
};

const GL_NO_ERROR: u32 = 0;

/// DRM fourcc `'AR24'` (`DRM_FORMAT_ARGB8888`): little-endian B,G,R,A
/// bytes — the layout Vulkan calls `B8G8R8A8_UNORM` and wgpu calls
/// `Bgra8Unorm`, and byte-identical to the macOS backend's BGRA8 output.
pub(crate) const FOURCC_ARGB8888: u32 = 0x3432_5241;
/// DRM fourcc `'XR24'` (`DRM_FORMAT_XRGB8888`) — the alpha-ignored
/// fallback for drivers that refuse a renderable linear AR24 bo. Same
/// byte layout; the X byte is what mpv wrote but consumers must not
/// trust it.
pub(crate) const FOURCC_XRGB8888: u32 = 0x3432_5258;

/// `DRM_FORMAT_MOD_LINEAR` — the only layout this backend produces (see
/// [`create_bo`]).
pub(crate) const MODIFIER_LINEAR: u64 = 0;

#[allow(non_snake_case)]
struct GlFns {
    GenTextures: unsafe extern "C" fn(i32, *mut u32),
    DeleteTextures: unsafe extern "C" fn(i32, *const u32),
    BindTexture: unsafe extern "C" fn(u32, u32),
    TexParameteri: unsafe extern "C" fn(u32, u32, i32),
    EGLImageTargetTexture2DOES: unsafe extern "C" fn(u32, *mut c_void),
    GenFramebuffers: unsafe extern "C" fn(i32, *mut u32),
    DeleteFramebuffers: unsafe extern "C" fn(i32, *const u32),
    BindFramebuffer: unsafe extern "C" fn(u32, u32),
    FramebufferTexture2D: unsafe extern "C" fn(u32, u32, u32, u32, i32),
    CheckFramebufferStatus: unsafe extern "C" fn(u32) -> u32,
    GetError: unsafe extern "C" fn() -> u32,
    Finish: unsafe extern "C" fn(),
}

fn egl_error(what: &str) -> Error {
    let code = unsafe { eglGetError() };
    Error::ExportSetup(format!("{what}: EGL error {code:#06x}"))
}

/// Load one entry point through [`gl_proc_address`] as the given fn
/// type, erroring on NULL.
macro_rules! load_fn {
    ($name:literal as $ty:ty) => {{
        let ptr = gl_proc_address($name);
        if ptr.is_null() {
            return Err(Error::ExportSetup(
                concat!("required GL/EGL entry point missing: ", $name).into(),
            ));
        }
        // SAFETY: resolved from the live EGL/GL implementation under the
        // name whose documented signature `$ty` spells.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
    }};
}

/// Open one DRM render node (`/dev/dri/renderD<minor>`) — render nodes
/// need no display server, no DRM master, and no seat, which is what
/// keeps this backend usable from any session that can see the GPU at
/// all.
fn open_node(minor: u32) -> Option<OwnedFd> {
    let path = format!("/dev/dri/renderD{minor}\0");
    let fd = unsafe {
        libc::open(
            path.as_ptr().cast::<c_char>(),
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return None;
    }
    // SAFETY: freshly opened, owned here.
    Some(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// RAII half-built pieces so `GlContext::new`'s error paths unwind
/// cleanly; on success they move into the finished context, whose field
/// order (display before device before fd) is the teardown order.
struct GbmHandle(*mut GbmDevice);
impl Drop for GbmHandle {
    fn drop(&mut self) {
        unsafe { gbm_device_destroy(self.0) };
    }
}
struct DisplayHandle(EGLDisplay);
impl Drop for DisplayHandle {
    fn drop(&mut self) {
        unsafe { eglTerminate(self.0) };
    }
}

fn display_has_extension(display: EGLDisplay, name: &str) -> bool {
    let exts = unsafe { eglQueryString(display, EGL_EXTENSIONS) };
    if exts.is_null() {
        return false;
    }
    let exts = unsafe { std::ffi::CStr::from_ptr(exts) }.to_string_lossy();
    exts.split_ascii_whitespace().any(|e| e == name)
}

/// Context ladder, mirroring the macOS profile ladder: desktop GL (the
/// driver's best compatibility version) first — it's what mpv is
/// happiest on — then GLES 3, then GLES 2.
fn create_context(display: EGLDisplay, no_config: bool) -> Result<EGLContext> {
    const ATTEMPTS: [(EGLenum, EGLint, &[EGLint]); 3] = [
        (EGL_OPENGL_API, EGL_OPENGL_BIT, &[EGL_NONE]),
        (
            EGL_OPENGL_ES_API,
            EGL_OPENGL_ES2_BIT,
            &[EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE],
        ),
        (
            EGL_OPENGL_ES_API,
            EGL_OPENGL_ES2_BIT,
            &[EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE],
        ),
    ];
    for (api, renderable_bit, ctx_attribs) in ATTEMPTS {
        if unsafe { eglBindAPI(api) } == 0 {
            continue;
        }
        let config = if no_config {
            EGL_NO_CONFIG
        } else {
            // Surface type 0: this context only ever renders to FBOs, so
            // any config of the right client API does.
            let attribs = [
                EGL_SURFACE_TYPE,
                0,
                EGL_RENDERABLE_TYPE,
                renderable_bit,
                EGL_NONE,
            ];
            let mut config: EGLConfig = std::ptr::null_mut();
            let mut count: EGLint = 0;
            let ok =
                unsafe { eglChooseConfig(display, attribs.as_ptr(), &mut config, 1, &mut count) };
            if ok == 0 || count < 1 {
                continue;
            }
            config
        };
        let ctx =
            unsafe { eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs.as_ptr()) };
        if !ctx.is_null() {
            return Ok(ctx);
        }
    }
    Err(egl_error("eglCreateContext (GL, GLES3, GLES2 all refused)"))
}

/// The hidden, windowless GL context the export render thread owns:
/// EGL on the GBM platform over a render node, made current
/// surfacelessly. Created, used, and dropped on that one thread.
pub(crate) struct GlContext {
    context: EGLContext,
    // Teardown order = declaration order (after `Drop` unbinds and
    // destroys the context): terminate the display, destroy the GBM
    // device, close the node.
    display: DisplayHandle,
    gbm: GbmHandle,
    _drm: OwnedFd,
    egl_create_image: PfnEglCreateImage,
    egl_destroy_image: PfnEglDestroyImage,
    fns: GlFns,
}

impl GlContext {
    fn on_node(drm: OwnedFd) -> Result<Self> {
        let gbm = unsafe { gbm_create_device(drm.as_raw_fd()) };
        if gbm.is_null() {
            return Err(Error::ExportSetup(
                "gbm_create_device failed on the render node".into(),
            ));
        }
        let gbm = GbmHandle(gbm);
        let display = unsafe {
            eglGetPlatformDisplay(EGL_PLATFORM_GBM, gbm.0.cast::<c_void>(), std::ptr::null())
        };
        if display == EGL_NO_DISPLAY {
            return Err(egl_error("eglGetPlatformDisplay(GBM)"));
        }
        if unsafe { eglInitialize(display, std::ptr::null_mut(), std::ptr::null_mut()) } == 0 {
            return Err(egl_error("eglInitialize"));
        }
        let display = DisplayHandle(display);
        for required in [
            "EGL_KHR_image_base",
            "EGL_EXT_image_dma_buf_import",
            "EGL_KHR_surfaceless_context",
        ] {
            if !display_has_extension(display.0, required) {
                return Err(Error::ExportSetup(format!(
                    "EGL display lacks {required} (driver too old for DMA-BUF export)"
                )));
            }
        }
        // Entry points load before the context exists on purpose:
        // `eglGetProcAddress` returns context-independent dispatch
        // pointers (calling them is what needs currency), and with the
        // loads done first, context creation is the last fallible step —
        // every error path unwinds through the RAII handles alone.
        let egl_create_image = load_fn!("eglCreateImageKHR" as PfnEglCreateImage);
        let egl_destroy_image = load_fn!("eglDestroyImageKHR" as PfnEglDestroyImage);
        let fns = GlFns {
            GenTextures: load_fn!("glGenTextures" as unsafe extern "C" fn(i32, *mut u32)),
            DeleteTextures: load_fn!("glDeleteTextures" as unsafe extern "C" fn(i32, *const u32)),
            BindTexture: load_fn!("glBindTexture" as unsafe extern "C" fn(u32, u32)),
            TexParameteri: load_fn!("glTexParameteri" as unsafe extern "C" fn(u32, u32, i32)),
            EGLImageTargetTexture2DOES: load_fn!(
                "glEGLImageTargetTexture2DOES" as unsafe extern "C" fn(u32, *mut c_void)
            ),
            GenFramebuffers: load_fn!("glGenFramebuffers" as unsafe extern "C" fn(i32, *mut u32)),
            DeleteFramebuffers: load_fn!(
                "glDeleteFramebuffers" as unsafe extern "C" fn(i32, *const u32)
            ),
            BindFramebuffer: load_fn!("glBindFramebuffer" as unsafe extern "C" fn(u32, u32)),
            FramebufferTexture2D: load_fn!(
                "glFramebufferTexture2D" as unsafe extern "C" fn(u32, u32, u32, u32, i32)
            ),
            CheckFramebufferStatus: load_fn!(
                "glCheckFramebufferStatus" as unsafe extern "C" fn(u32) -> u32
            ),
            GetError: load_fn!("glGetError" as unsafe extern "C" fn() -> u32),
            Finish: load_fn!("glFinish" as unsafe extern "C" fn()),
        };
        let no_config = display_has_extension(display.0, "EGL_KHR_no_config_context");
        let context = create_context(display.0, no_config)?;
        Ok(Self {
            context,
            display,
            gbm,
            _drm: drm,
            egl_create_image,
            egl_destroy_image,
            fns,
        })
    }
}

impl ExportBackend for GlContext {
    type Buffer = SurfaceBuffer;

    /// The FBO color format reported to mpv (`OpenGlFbo::internal_format`).
    /// A hint only — the real storage layout is fixed by the DMA-BUF's DRM
    /// fourcc, which GL learns through the EGLImage.
    const FBO_INTERNAL_FORMAT: i32 = GL_RGBA8 as i32;

    /// Bring the backend up on the first *usable* render node — a node
    /// that opens is not enough on hybrid-GPU boxes, where e.g.
    /// `renderD128` may lack EGL-on-GBM/dma-buf support while
    /// `renderD129` is fully capable, so every setup failure falls
    /// through to the next node (the wlroots/Mesa device-selection
    /// discipline) rather than failing the attach.
    fn new() -> Result<Self> {
        let mut last_error = None;
        for minor in 128..192 {
            let Some(drm) = open_node(minor) else {
                continue;
            };
            match Self::on_node(drm) {
                Ok(gl) => return Ok(gl),
                Err(e) => {
                    tracing::debug!("export: render node renderD{minor} unusable: {e}");
                    last_error = Some(e);
                }
            }
        }
        Err(match last_error {
            Some(Error::ExportSetup(msg)) => {
                Error::ExportSetup(format!("no usable DRM render node (last tried: {msg})"))
            }
            Some(e) => e,
            None => Error::ExportSetup(
                "no DRM render node under /dev/dri (no GPU, or no permission — user not in the render/video group?)"
                    .into(),
            ),
        })
    }

    fn make_current(&self) -> Result<()> {
        // Surfaceless: no EGLSurface exists anywhere in this backend;
        // rendering targets are FBOs (EGL_KHR_surfaceless_context,
        // presence checked at init).
        let ok =
            unsafe { eglMakeCurrent(self.display.0, EGL_NO_SURFACE, EGL_NO_SURFACE, self.context) };
        if ok == 0 {
            return Err(egl_error("eglMakeCurrent (surfaceless)"));
        }
        Ok(())
    }

    /// Nothing to hand over before mpv renders: a dmabuf-backed GL
    /// texture is always the render thread's to write. (Windows, whose
    /// exportable storage lives behind a `WGL_NV_DX_interop2` lock, is
    /// why the cross-platform seam has this call at all.)
    fn begin_render(&self, _buffer: &SurfaceBuffer) {}

    /// Publish barrier before a DMA-BUF is consumed by another API.
    /// Unlike IOSurface's flush-coherency contract on macOS, a DMA-BUF
    /// carries no cross-API ordering guarantee a mere `glFlush` would
    /// satisfy — Vulkan consumers don't participate in the kernel's
    /// implicit sync. Until explicit sync-fd export lands (roadmap),
    /// completion is guaranteed the blunt way: `glFinish` on this
    /// dedicated thread, so a published frame's pixels are already on
    /// the bus when the shell sees it and no consumer-side wait exists.
    fn publish_barrier(&self, _buffer: &SurfaceBuffer) {
        unsafe { (self.fns.Finish)() };
    }

    fn proc_address(name: &str) -> *mut c_void {
        gl_proc_address(name)
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            eglMakeCurrent(
                self.display.0,
                EGL_NO_SURFACE,
                EGL_NO_SURFACE,
                EGL_NO_CONTEXT,
            );
            eglDestroyContext(self.display.0, self.context);
        }
        // RAII fields finish the teardown in declaration order.
    }
}

/// One pool entry: a linear BGRA8 DMA-BUF wrapped (via EGLImage) in a GL
/// texture and a framebuffer mpv renders into. The fd half travels
/// across threads inside [`ExportedFrame`](crate::ExportedFrame); the
/// GL/EGL names are handles only the render thread ever dereferences.
pub(crate) struct SurfaceBuffer {
    fd: OwnedFd,
    image: EGLImage,
    texture: u32,
    fbo: u32,
    stride: u32,
    fourcc: u32,
    width: u32,
    height: u32,
}

// SAFETY: the fd is a kernel object, usable and closable from any
// thread. `image` and the GL names are plain handles here; every call
// against them (creation, deletion, mpv's renders) happens on the render
// thread — see the module docs' thread discipline. `Drop` touches only
// the fd.
unsafe impl Send for SurfaceBuffer {}
unsafe impl Sync for SurfaceBuffer {}

/// Allocate the exportable bo: `GBM_BO_USE_LINEAR` on purpose. Linear
/// costs a little GPU bandwidth but removes the entire DRM-modifier
/// negotiation problem from the seam — consumers import with
/// `DRM_FORMAT_MOD_LINEAR` unconditionally, and [`SurfaceBuffer::copy_pixels`]
/// can be a plain mmap. A modifier-negotiated tier can arrive later as
/// an additive `ExportOptions` knob. AR24 first, XR24 for drivers that
/// won't render to linear AR24.
fn create_bo(gbm: *mut GbmDevice, width: u32, height: u32) -> Result<(*mut GbmBo, u32)> {
    for fourcc in [FOURCC_ARGB8888, FOURCC_XRGB8888] {
        let bo = unsafe {
            gbm_bo_create(
                gbm,
                width,
                height,
                fourcc,
                GBM_USE_RENDERING | GBM_USE_LINEAR,
            )
        };
        if !bo.is_null() {
            return Ok((bo, fourcc));
        }
    }
    Err(Error::ExportSetup(format!(
        "gbm_bo_create failed for {width}x{height} linear AR24/XR24"
    )))
}

impl ExportBuffer for SurfaceBuffer {
    type Backend = GlContext;

    /// Create a `width`×`height` buffer. Render thread only, GL context
    /// current.
    fn new(gl: &GlContext, width: u32, height: u32) -> Result<Self> {
        let (bo, fourcc) = create_bo(gl.gbm.0, width, height)?;
        let raw_fd = unsafe { gbm_bo_get_fd(bo) };
        if raw_fd < 0 {
            unsafe { gbm_bo_destroy(bo) };
            return Err(Error::ExportSetup("gbm_bo_get_fd failed".into()));
        }
        // SAFETY: gbm_bo_get_fd returns a new fd owned by the caller.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let stride = unsafe { gbm_bo_get_stride(bo) };
        // Import our own export: the EGLImage takes its own reference on
        // the dmabuf (the fd stays ours to keep and close), after which
        // the bo handle is redundant — fd + image pin the memory.
        let attribs = [
            EGL_WIDTH,
            width as EGLint,
            EGL_HEIGHT,
            height as EGLint,
            EGL_LINUX_DRM_FOURCC,
            fourcc as EGLint,
            EGL_DMA_BUF_PLANE0_FD,
            fd.as_raw_fd(),
            EGL_DMA_BUF_PLANE0_OFFSET,
            0,
            EGL_DMA_BUF_PLANE0_PITCH,
            stride as EGLint,
            EGL_NONE,
        ];
        let image = unsafe {
            (gl.egl_create_image)(
                gl.display.0,
                EGL_NO_CONTEXT,
                EGL_LINUX_DMA_BUF,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        unsafe { gbm_bo_destroy(bo) };
        if image.is_null() {
            return Err(egl_error("eglCreateImageKHR(DMA_BUF)"));
        }

        let f = &gl.fns;
        let mut texture: u32 = 0;
        unsafe {
            (f.GetError)(); // clear stale error state
            (f.GenTextures)(1, &mut texture);
            (f.BindTexture)(GL_TEXTURE_2D, texture);
            (f.EGLImageTargetTexture2DOES)(GL_TEXTURE_2D, image);
        }
        let bind_error = unsafe { (f.GetError)() };
        unsafe {
            (f.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            (f.TexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            (f.BindTexture)(GL_TEXTURE_2D, 0);
        }
        if bind_error != GL_NO_ERROR {
            unsafe {
                (f.DeleteTextures)(1, &texture);
                (gl.egl_destroy_image)(gl.display.0, image);
            }
            return Err(Error::ExportSetup(format!(
                "glEGLImageTargetTexture2DOES failed (GL error {bind_error:#06x})"
            )));
        }
        let mut fbo: u32 = 0;
        let status = unsafe {
            (f.GenFramebuffers)(1, &mut fbo);
            (f.BindFramebuffer)(GL_FRAMEBUFFER, fbo);
            (f.FramebufferTexture2D)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                texture,
                0,
            );
            let status = (f.CheckFramebufferStatus)(GL_FRAMEBUFFER);
            (f.BindFramebuffer)(GL_FRAMEBUFFER, 0);
            status
        };
        if status != GL_FRAMEBUFFER_COMPLETE {
            unsafe {
                (f.DeleteFramebuffers)(1, &fbo);
                (f.DeleteTextures)(1, &texture);
                (gl.egl_destroy_image)(gl.display.0, image);
            }
            return Err(Error::ExportSetup(format!(
                "DMA-BUF framebuffer incomplete (status {status:#x})"
            )));
        }
        Ok(Self {
            fd,
            image,
            texture,
            fbo,
            stride,
            fourcc,
            width,
            height,
        })
    }

    fn fbo(&self) -> u32 {
        self.fbo
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Delete the GL texture/framebuffer names and the EGLImage. Render
    /// thread only, GL context current; the fd itself is closed by
    /// `Drop` (any thread). Buffers retired anywhere else simply skip
    /// this — their GL names and image die with the render thread's
    /// context and display.
    fn delete_gl(&mut self, gl: &GlContext) {
        let f = &gl.fns;
        unsafe {
            if self.fbo != 0 {
                (f.DeleteFramebuffers)(1, &self.fbo);
                self.fbo = 0;
            }
            if self.texture != 0 {
                (f.DeleteTextures)(1, &self.texture);
                self.texture = 0;
            }
            if !self.image.is_null() {
                (gl.egl_destroy_image)(gl.display.0, self.image);
                self.image = std::ptr::null_mut();
            }
        }
    }

    /// Copy the buffer's pixels out as tightly packed BGRA rows
    /// (row 0 = top). Any thread: this maps the dmabuf fd directly (the
    /// buffer is linear by construction) with the kernel's
    /// `DMA_BUF_IOCTL_SYNC` bracketing for CPU cache coherency — no GL,
    /// no GBM, no EGL, so it works even after the render context is
    /// gone. Returns zeroed pixels if the exporter refuses CPU mapping
    /// (mirroring the macOS lock-failure behavior).
    fn copy_pixels(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let stride = self.stride as usize;
        let len = stride * h;
        if len == 0 {
            return vec![0u8; w * h * 4];
        }
        unsafe {
            let base = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                self.fd.as_raw_fd(),
                0,
            );
            if base == libc::MAP_FAILED {
                return vec![0u8; w * h * 4];
            }
            // Best-effort sync bracket: exporters without the ioctl are
            // coherent-mapping ones, so failure is ignorable.
            dma_buf_sync(&self.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
            // SAFETY: `base` maps `len == stride * h` readable bytes with
            // `stride >= w * 4`.
            let out = super::pack_bgra_rows(base.cast::<u8>(), stride, w, h);
            dma_buf_sync(&self.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
            libc::munmap(base, len);
            out
        }
    }
}

/// mpv's GL loader for the hidden context (and the module's own
/// `load_fn!`): EGL 1.5's `eglGetProcAddress` resolves client-API and EGL
/// entry points alike (the README's Linux loader note), with a `dlsym`
/// fallback for the rare libEGL that still won't return core GL symbols.
/// [`ExportBackend::proc_address`](super::ExportBackend::proc_address)
/// delegates here.
fn gl_proc_address(name: &str) -> *mut c_void {
    let Ok(cname) = CString::new(name) else {
        return std::ptr::null_mut();
    };
    let ptr = unsafe { eglGetProcAddress(cname.as_ptr()) };
    if !ptr.is_null() {
        return ptr;
    }
    unsafe { libc::dlsym(libc::RTLD_DEFAULT, cname.as_ptr()) }
}

/// Linux's platform-native export accessors: the DMA-BUF handle and its
/// layout. Deliberately outside the [`ExportBuffer`](super::ExportBuffer)
/// contract — each platform's export currency differs — and used only by
/// the Linux wgpu import and the frame's public dmabuf accessors.
impl SurfaceBuffer {
    pub(crate) fn dma_buf_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    pub(crate) fn stride(&self) -> u32 {
        self.stride
    }

    pub(crate) fn fourcc(&self) -> u32 {
        self.fourcc
    }
}

// No Drop needed beyond the derived one: `OwnedFd` closes the fd; GL
// names and the EGLImage are handled per `delete_gl`'s contract.

const DMA_BUF_SYNC_READ: u64 = 1 << 0;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 1 << 2;
/// `_IOW('b', 0, struct dma_buf_sync)` — `struct dma_buf_sync` is a
/// single `__u64 flags`.
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200;

fn dma_buf_sync(fd: &OwnedFd, flags: u64) {
    let sync = flags;
    unsafe {
        libc::ioctl(fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC, &raw const sync);
    }
}
