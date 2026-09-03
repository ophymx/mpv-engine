//! Render seam — OpenGL and software backends, via `rsmpv::sys`
//! directly.
//!
//! This bypasses `rsmpv`'s safe `RenderContext` because that type borrows
//! the `Mpv` (`RenderContext<'mpv>`), so it cannot be stored beside its
//! handle in one owning struct (self-referential). The raw context plus an
//! explicit `Drop` reproduces the safety property that actually matters —
//! free the render context before the handle — without the borrow.
//!
//! TODO(upstream): an owned (non-borrowing) `RenderContext` in rsmpv —
//! e.g. one holding the raw handle pointer the way this module does —
//! would let the engine adopt the safe type and shrink this module to the
//! backend-specific param plumbing.
//!
//! Threading contract: the update callback fires on **mpv's render
//! thread** (hence `Send`), and also *synchronously during registration* —
//! consumers must be re-entrant-safe at attach time. `render` must be
//! called with the target GL context current, and the context must also be
//! current when this struct drops: `mpv_render_context_free` tears down GL
//! objects, and freeing without the right context current leaks them into
//! whatever context *is* current (in GTK that manifested as whole-window
//! rendering artifacts after the player page was popped).

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use rsmpv::sys;

use crate::error::{Error, Result};

/// Resolves GL symbols for mpv. Called during context creation (and
/// possibly later), so it must stay alive for the context's lifetime.
pub type ProcAddressFn = Box<dyn Fn(&str) -> *mut c_void + Send + Sync>;
type UpdateFn = Box<dyn Fn() + Send>;

/// The one attached render backend. Backends share the engine's single
/// slot so `AlreadyAttached` and `detach_render` behave uniformly —
/// mpv allows one render context per handle regardless of type.
pub(crate) enum RenderBackend {
    Gl(GlRender),
    Sw(SwRender),
}

pub(crate) struct GlRender {
    ctx: *mut sys::mpv_render_context,
    // Both closures are double-boxed: the outer Box gives the *inner* fat
    // Box a stable thin address that the C thunks cast back from. They are
    // referenced by mpv until `mpv_render_context_free` in `Drop`, which
    // runs before Rust frees the fields (drop glue order).
    _get_proc: Box<ProcAddressFn>,
    _on_update: Box<UpdateFn>,
}

// The context pointer is only handed to libmpv, which is internally
// synchronized; the closures are Send (+Sync for the proc lookup).
unsafe impl Send for GlRender {}
unsafe impl Sync for GlRender {}

impl GlRender {
    /// Create an OpenGL render context on `mpv` and register `on_update`.
    ///
    /// `on_update` fires once synchronously here (mpv's documented
    /// behavior) and afterwards from the render thread.
    pub(crate) fn create(
        mpv: *mut sys::mpv_handle,
        get_proc_address: ProcAddressFn,
        on_update: impl Fn() + Send + 'static,
    ) -> Result<Self> {
        let get_proc: Box<ProcAddressFn> = Box::new(get_proc_address);
        let on_update: Box<UpdateFn> = Box::new(Box::new(on_update));

        let mut init_params = sys::mpv_opengl_init_params {
            get_proc_address: Some(proc_thunk),
            get_proc_address_ctx: &*get_proc as *const ProcAddressFn as *mut c_void,
        };
        let mut params = [
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_API_TYPE,
                data: sys::MPV_RENDER_API_TYPE_OPENGL.as_ptr() as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_OPENGL_INIT_PARAMS,
                data: &mut init_params as *mut _ as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_INVALID,
                data: ptr::null_mut(),
            },
        ];

        let mut ctx: *mut sys::mpv_render_context = ptr::null_mut();
        let err = unsafe { sys::mpv_render_context_create(&mut ctx, mpv, params.as_mut_ptr()) };
        if err < 0 {
            return Err(Error::Render(err));
        }

        unsafe {
            sys::mpv_render_context_set_update_callback(
                ctx,
                Some(update_thunk),
                &*on_update as *const UpdateFn as *mut c_void,
            );
        }

        Ok(Self {
            ctx,
            _get_proc: get_proc,
            _on_update: on_update,
        })
    }

    /// Draw the current frame into `fbo` (`0` = default framebuffer).
    /// `flip_y` handles targets with a flipped origin (e.g. GTK's GLArea).
    pub(crate) fn render(&self, fbo: i32, w: i32, h: i32, flip_y: bool) -> Result<()> {
        let mut gl_fbo = sys::mpv_opengl_fbo {
            fbo,
            w,
            h,
            internal_format: 0,
        };
        let mut flip: c_int = flip_y as c_int;
        let mut params = [
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_OPENGL_FBO,
                data: &mut gl_fbo as *mut _ as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_FLIP_Y,
                data: &mut flip as *mut c_int as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_INVALID,
                data: ptr::null_mut(),
            },
        ];
        let err = unsafe { sys::mpv_render_context_render(self.ctx, params.as_mut_ptr()) };
        if err < 0 {
            return Err(Error::Render(err));
        }
        Ok(())
    }
}

impl Drop for GlRender {
    fn drop(&mut self) {
        // Clear the callback first so the render thread can't fire into a
        // half-dead struct, then free. See the module doc for the
        // GL-context-current requirement.
        unsafe {
            sys::mpv_render_context_set_update_callback(self.ctx, None, ptr::null_mut());
            sys::mpv_render_context_free(self.ctx);
        }
    }
}

/// Software rendering (`MPV_RENDER_API_TYPE_SW`): mpv draws the frame
/// into a caller-provided RGBA buffer. No GL anywhere — no
/// context-current requirements for rendering *or* teardown, so unlike
/// [`GlRender`] this backend is safe to drop from any thread.
pub(crate) struct SwRender {
    ctx: *mut sys::mpv_render_context,
    // Double-boxed for a stable thin address; see `GlRender`.
    _on_update: Box<UpdateFn>,
}

// The context pointer is only handed to libmpv, which is internally
// synchronized; the closure is Send.
unsafe impl Send for SwRender {}
unsafe impl Sync for SwRender {}

impl SwRender {
    /// Create a software render context on `mpv` and register
    /// `on_update` (same contract as the GL backend: fires once
    /// synchronously here, afterwards from mpv's render thread).
    pub(crate) fn create(
        mpv: *mut sys::mpv_handle,
        on_update: impl Fn() + Send + 'static,
    ) -> Result<Self> {
        let on_update: Box<UpdateFn> = Box::new(Box::new(on_update));

        let mut params = [
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_API_TYPE,
                data: sys::MPV_RENDER_API_TYPE_SW.as_ptr() as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_INVALID,
                data: ptr::null_mut(),
            },
        ];

        let mut ctx: *mut sys::mpv_render_context = ptr::null_mut();
        let err = unsafe { sys::mpv_render_context_create(&mut ctx, mpv, params.as_mut_ptr()) };
        if err < 0 {
            return Err(Error::Render(err));
        }

        unsafe {
            sys::mpv_render_context_set_update_callback(
                ctx,
                Some(update_thunk),
                &*on_update as *const UpdateFn as *mut c_void,
            );
        }

        Ok(Self {
            ctx,
            _on_update: on_update,
        })
    }

    /// Render the current frame as RGBA8 into `buf`, resizing it to
    /// `w * h * 4`.
    pub(crate) fn render(&self, w: i32, h: i32, buf: &mut Vec<u8>) -> Result<()> {
        // Nothing to draw for empty dimensions — and a negative `w` would
        // sign-extend through `as usize` into a huge allocation (panic),
        // with the multiply itself able to overflow on 32-bit targets.
        // The GL path is immune (mpv validates its ints); this path
        // allocates first, so it must guard. No-op mirrors the
        // render-before-attach philosophy.
        let (Ok(uw), Ok(uh)) = (usize::try_from(w), usize::try_from(h)) else {
            return Ok(());
        };
        let Some(len) = uw.checked_mul(uh).and_then(|p| p.checked_mul(4)) else {
            return Ok(());
        };
        if len == 0 {
            return Ok(());
        }
        let mut size: [c_int; 2] = [w, h];
        let mut stride: usize = uw * 4;
        buf.resize(len, 0);
        let mut params = [
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_SW_SIZE,
                data: size.as_mut_ptr() as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_SW_FORMAT,
                data: c"rgb0".as_ptr() as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_SW_STRIDE,
                data: &mut stride as *mut usize as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_SW_POINTER,
                data: buf.as_mut_ptr() as *mut c_void,
            },
            sys::mpv_render_param {
                type_: sys::MPV_RENDER_PARAM_INVALID,
                data: ptr::null_mut(),
            },
        ];
        let err = unsafe { sys::mpv_render_context_render(self.ctx, params.as_mut_ptr()) };
        if err < 0 {
            return Err(Error::Render(err));
        }
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

impl Drop for SwRender {
    fn drop(&mut self) {
        // Clear the callback first so the render thread can't fire into
        // a half-dead struct, then free. No GL objects involved, so no
        // context-current requirement here.
        unsafe {
            sys::mpv_render_context_set_update_callback(self.ctx, None, ptr::null_mut());
            sys::mpv_render_context_free(self.ctx);
        }
    }
}

unsafe extern "C" fn proc_thunk(ctx: *mut c_void, name: *const c_char) -> *mut c_void {
    let f = unsafe { &*(ctx as *const ProcAddressFn) };
    match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => f(s),
        Err(_) => ptr::null_mut(),
    }
}

unsafe extern "C" fn update_thunk(ctx: *mut c_void) {
    let f = unsafe { &*(ctx as *const UpdateFn) };
    f();
}
