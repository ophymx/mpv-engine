//! Raw macOS plumbing for the exported-frame backend: a hidden CGL
//! context, IOSurface-backed GL framebuffers, and the OpenGL.framework
//! symbol loader mpv renders through. The three frameworks are linked
//! directly — no Rust dependency enters the tree for this.
//!
//! Thread discipline: everything that touches GL — context creation and
//! currency, [`SurfaceBuffer::new`], [`SurfaceBuffer::delete_gl`] — runs
//! only on the engine's export render thread (the module's only GL
//! caller), so GL's thread rules are upheld by construction rather than
//! by types. The IOSurface side is thread-safe CoreFoundation — that's
//! exactly why it is the export currency.

use std::ffi::{CString, c_char, c_void};

use crate::error::{Error, Result};

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFIndex = isize;

/// `kCFNumberSInt32Type`.
const CF_NUMBER_SINT32: CFIndex = 3;

/// Opaque stand-in for `CFDictionaryKey/ValueCallBacks`; only the
/// addresses of the exported `kCFType*` tables are ever used.
#[repr(C)]
struct CFDictionaryCallBacks {
    _opaque: [u8; 0],
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        num_values: CFIndex,
        key_callbacks: *const CFDictionaryCallBacks,
        value_callbacks: *const CFDictionaryCallBacks,
    ) -> CFDictionaryRef;
    fn CFNumberCreate(
        allocator: *const c_void,
        the_type: CFIndex,
        value_ptr: *const c_void,
    ) -> CFTypeRef;
    static kCFTypeDictionaryKeyCallBacks: CFDictionaryCallBacks;
    static kCFTypeDictionaryValueCallBacks: CFDictionaryCallBacks;
}

/// A retained `IOSurfaceRef` (as the raw pointer consumers hand to
/// Metal's `newTextureWithDescriptor:iosurface:plane:`).
pub(crate) type IOSurfaceRef = *mut c_void;

/// `kIOSurfaceLockReadOnly`.
const IOSURFACE_LOCK_READ_ONLY: u32 = 1;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLock(buffer: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(buffer: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(buffer: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetBytesPerRow(buffer: IOSurfaceRef) -> usize;
    static kIOSurfaceWidth: CFStringRef;
    static kIOSurfaceHeight: CFStringRef;
    static kIOSurfaceBytesPerElement: CFStringRef;
    static kIOSurfacePixelFormat: CFStringRef;
}

type CGLPixelFormatObj = *mut c_void;
type CGLContextObj = *mut c_void;

/// `kCGLPFAOpenGLProfile`.
const CGL_PFA_OPENGL_PROFILE: i32 = 99;
/// `kCGLPFAAllowOfflineRenderers` — lets pixel-format selection succeed
/// on renderers with no display attached (headless boxes, CI).
const CGL_PFA_ALLOW_OFFLINE: i32 = 96;
/// `kCGLOGLPVersion_GL4_Core` / `_GL3_Core` / `_Legacy`, tried in that
/// order — mpv is happiest on the newest core profile the box offers.
const CGL_PROFILE_GL4_CORE: i32 = 0x4100;
const CGL_PROFILE_GL3_CORE: i32 = 0x3200;
const CGL_PROFILE_LEGACY: i32 = 0x1000;

// FBO/framebuffer GL enums shared with the other backends.
use super::gl_consts::{GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_FRAMEBUFFER_COMPLETE, GL_RGBA8};

const GL_TEXTURE_RECTANGLE: u32 = 0x84F5;
const GL_RGBA: u32 = 0x1908;
const GL_BGRA: u32 = 0x80E1;
const GL_UNSIGNED_INT_8_8_8_8_REV: u32 = 0x8367;

/// The FBO color format reported to mpv (`OpenGlFbo::internal_format`).
pub(crate) const FBO_INTERNAL_FORMAT: i32 = GL_RGBA8 as i32;

#[link(name = "OpenGL", kind = "framework")]
unsafe extern "C" {
    fn CGLChoosePixelFormat(
        attribs: *const i32,
        pix: *mut CGLPixelFormatObj,
        npix: *mut i32,
    ) -> i32;
    fn CGLDestroyPixelFormat(pix: CGLPixelFormatObj) -> i32;
    fn CGLCreateContext(
        pix: CGLPixelFormatObj,
        share: CGLContextObj,
        ctx: *mut CGLContextObj,
    ) -> i32;
    fn CGLDestroyContext(ctx: CGLContextObj) -> i32;
    fn CGLSetCurrentContext(ctx: CGLContextObj) -> i32;
    fn CGLGetCurrentContext() -> CGLContextObj;
    fn CGLErrorString(code: i32) -> *const c_char;
    fn CGLTexImageIOSurface2D(
        ctx: CGLContextObj,
        target: u32,
        internal_format: u32,
        width: i32,
        height: i32,
        format: u32,
        type_: u32,
        io_surface: IOSurfaceRef,
        plane: u32,
    ) -> i32;

    fn glGenTextures(n: i32, textures: *mut u32);
    fn glDeleteTextures(n: i32, textures: *const u32);
    fn glBindTexture(target: u32, texture: u32);
    fn glGenFramebuffers(n: i32, framebuffers: *mut u32);
    fn glDeleteFramebuffers(n: i32, framebuffers: *const u32);
    fn glBindFramebuffer(target: u32, framebuffer: u32);
    fn glFramebufferTexture2D(
        target: u32,
        attachment: u32,
        textarget: u32,
        texture: u32,
        level: i32,
    );
    fn glCheckFramebufferStatus(target: u32) -> u32;
    fn glFlush();
}

fn cgl_error(what: &str, code: i32) -> Error {
    let text = unsafe {
        let ptr = CGLErrorString(code);
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Error::ExportSetup(format!("{what}: CGL error {code} ({text})"))
}

/// The hidden, windowless GL context the export render thread owns.
/// Created, used, and dropped on that one thread.
pub(crate) struct GlContext {
    ctx: CGLContextObj,
}

impl GlContext {
    pub(crate) fn new() -> Result<Self> {
        for profile in [
            CGL_PROFILE_GL4_CORE,
            CGL_PROFILE_GL3_CORE,
            CGL_PROFILE_LEGACY,
        ] {
            let attribs = [CGL_PFA_OPENGL_PROFILE, profile, CGL_PFA_ALLOW_OFFLINE, 0];
            let mut pix: CGLPixelFormatObj = std::ptr::null_mut();
            let mut npix: i32 = 0;
            let code = unsafe { CGLChoosePixelFormat(attribs.as_ptr(), &mut pix, &mut npix) };
            if code != 0 || pix.is_null() {
                continue;
            }
            let mut ctx: CGLContextObj = std::ptr::null_mut();
            let code = unsafe { CGLCreateContext(pix, std::ptr::null_mut(), &mut ctx) };
            unsafe { CGLDestroyPixelFormat(pix) };
            if code == 0 && !ctx.is_null() {
                return Ok(Self { ctx });
            }
        }
        Err(Error::ExportSetup(
            "no CGL pixel format/context available (session without WindowServer/GPU access?)"
                .into(),
        ))
    }

    pub(crate) fn make_current(&self) -> Result<()> {
        let code = unsafe { CGLSetCurrentContext(self.ctx) };
        if code != 0 {
            return Err(cgl_error("CGLSetCurrentContext", code));
        }
        Ok(())
    }

    /// Nothing to hand over before mpv renders: an IOSurface-backed GL
    /// texture is always the render thread's to write. (Windows, whose
    /// exportable storage lives behind a `WGL_NV_DX_interop2` lock, is
    /// why the cross-platform seam has this call at all.)
    pub(crate) fn begin_render(&self, _buffer: &SurfaceBuffer) {}

    /// Publish barrier before an IOSurface is sampled from another API:
    /// IOSurface guarantees cross-API coherency only once the producing
    /// GL context flushes (a full `glFinish` stall is *not* required —
    /// don't "strengthen" this; the Linux and Windows backends finish
    /// for reasons of their own).
    pub(crate) fn publish_barrier(&self, _buffer: &SurfaceBuffer) {
        unsafe { glFlush() };
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            if CGLGetCurrentContext() == self.ctx {
                CGLSetCurrentContext(std::ptr::null_mut());
            }
            CGLDestroyContext(self.ctx);
        }
    }
}

/// One pool entry: a BGRA8 IOSurface wrapped in a GL rectangle texture
/// and a framebuffer mpv renders into. The IOSurface half travels across
/// threads inside [`ExportedFrame`](crate::ExportedFrame); the GL names
/// are integers only the render thread ever dereferences.
pub(crate) struct SurfaceBuffer {
    surface: IOSurfaceRef,
    texture: u32,
    fbo: u32,
    width: u32,
    height: u32,
}

// SAFETY: IOSurface is a thread-safe CoreFoundation object (lock, read,
// release from any thread). The GL names are plain integers here; every
// GL call against them (creation, deletion, mpv's renders) happens on
// the render thread — see the module docs' thread discipline.
unsafe impl Send for SurfaceBuffer {}
unsafe impl Sync for SurfaceBuffer {}

impl SurfaceBuffer {
    /// Create a `width`×`height` buffer. Render thread only, GL context
    /// current (`_gl` is the cross-platform signature; CGL resolves the
    /// current context itself).
    pub(crate) fn new(_gl: &GlContext, width: u32, height: u32) -> Result<Self> {
        let surface = create_iosurface(width, height)?;
        let cgl = unsafe { CGLGetCurrentContext() };
        let mut texture: u32 = 0;
        unsafe {
            glGenTextures(1, &mut texture);
            glBindTexture(GL_TEXTURE_RECTANGLE, texture);
        }
        // Sized internal format first (what a core profile wants); the
        // unsized spelling is the fallback older drivers accepted.
        let mut code = unsafe {
            CGLTexImageIOSurface2D(
                cgl,
                GL_TEXTURE_RECTANGLE,
                GL_RGBA8,
                width as i32,
                height as i32,
                GL_BGRA,
                GL_UNSIGNED_INT_8_8_8_8_REV,
                surface,
                0,
            )
        };
        if code != 0 {
            code = unsafe {
                CGLTexImageIOSurface2D(
                    cgl,
                    GL_TEXTURE_RECTANGLE,
                    GL_RGBA,
                    width as i32,
                    height as i32,
                    GL_BGRA,
                    GL_UNSIGNED_INT_8_8_8_8_REV,
                    surface,
                    0,
                )
            };
        }
        unsafe { glBindTexture(GL_TEXTURE_RECTANGLE, 0) };
        if code != 0 {
            unsafe {
                glDeleteTextures(1, &texture);
                CFRelease(surface);
            }
            return Err(cgl_error("CGLTexImageIOSurface2D", code));
        }
        let mut fbo: u32 = 0;
        let status = unsafe {
            glGenFramebuffers(1, &mut fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_RECTANGLE,
                texture,
                0,
            );
            let status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            status
        };
        if status != GL_FRAMEBUFFER_COMPLETE {
            unsafe {
                glDeleteFramebuffers(1, &fbo);
                glDeleteTextures(1, &texture);
                CFRelease(surface);
            }
            return Err(Error::ExportSetup(format!(
                "IOSurface framebuffer incomplete (status {status:#x})"
            )));
        }
        Ok(Self {
            surface,
            texture,
            fbo,
            width,
            height,
        })
    }

    pub(crate) fn fbo(&self) -> u32 {
        self.fbo
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn io_surface(&self) -> IOSurfaceRef {
        self.surface
    }

    /// Delete the GL texture/framebuffer names. Render thread only, GL
    /// context current; the IOSurface itself is released by `Drop` (any
    /// thread). Buffers retired anywhere else simply skip this — their
    /// GL names die with the render thread's context.
    pub(crate) fn delete_gl(&mut self, _gl: &GlContext) {
        unsafe {
            if self.fbo != 0 {
                glDeleteFramebuffers(1, &self.fbo);
                self.fbo = 0;
            }
            if self.texture != 0 {
                glDeleteTextures(1, &self.texture);
                self.texture = 0;
            }
        }
    }

    /// Copy the surface's pixels out as tightly packed BGRA rows
    /// (row 0 = top). Any thread.
    pub(crate) fn copy_pixels(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        unsafe {
            if IOSurfaceLock(self.surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut()) != 0 {
                return vec![0u8; w * h * 4];
            }
            let base = IOSurfaceGetBaseAddress(self.surface) as *const u8;
            let stride = IOSurfaceGetBytesPerRow(self.surface);
            // SAFETY: the surface is locked; when non-null, `base` covers
            // `stride * h` readable bytes with `stride >= w * 4`.
            let out = if base.is_null() {
                vec![0u8; w * h * 4]
            } else {
                super::pack_bgra_rows(base, stride, w, h)
            };
            IOSurfaceUnlock(self.surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut());
            out
        }
    }
}

impl Drop for SurfaceBuffer {
    fn drop(&mut self) {
        // Only the IOSurface needs an explicit release here; GL names
        // are handled per `delete_gl`'s contract.
        unsafe { CFRelease(self.surface) };
    }
}

fn create_iosurface(width: u32, height: u32) -> Result<IOSurfaceRef> {
    /// fourcc `'BGRA'` (`kCVPixelFormatType_32BGRA`) — the interchange
    /// format every macOS consumer (Metal `bgra8Unorm`, CoreVideo,
    /// CoreImage) takes without conversion.
    const PIXEL_FORMAT_BGRA: i32 = 0x42475241;
    let (w, h) = (width as i32, height as i32);
    let bytes_per_element: i32 = 4;
    let pixel_format: i32 = PIXEL_FORMAT_BGRA;
    unsafe {
        let values = [
            CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                (&raw const w).cast::<c_void>(),
            ),
            CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                (&raw const h).cast::<c_void>(),
            ),
            CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                (&raw const bytes_per_element).cast::<c_void>(),
            ),
            CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                (&raw const pixel_format).cast::<c_void>(),
            ),
        ];
        let keys = [
            kIOSurfaceWidth as CFTypeRef,
            kIOSurfaceHeight as CFTypeRef,
            kIOSurfaceBytesPerElement as CFTypeRef,
            kIOSurfacePixelFormat as CFTypeRef,
        ];
        let dict = CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            keys.len() as CFIndex,
            &raw const kCFTypeDictionaryKeyCallBacks,
            &raw const kCFTypeDictionaryValueCallBacks,
        );
        for value in values {
            CFRelease(value);
        }
        if dict.is_null() {
            return Err(Error::ExportSetup("CFDictionaryCreate failed".into()));
        }
        let surface = IOSurfaceCreate(dict);
        CFRelease(dict);
        if surface.is_null() {
            return Err(Error::ExportSetup(format!(
                "IOSurfaceCreate failed for {width}x{height}"
            )));
        }
        Ok(surface)
    }
}

/// mpv's GL loader for the hidden context: OpenGL.framework is linked
/// into the process, so every GL entry point resolves through a plain
/// `dlsym` — no windowing-toolkit loader involved.
pub(crate) fn gl_proc_address(name: &str) -> *mut c_void {
    let Ok(cname) = CString::new(name) else {
        return std::ptr::null_mut();
    };
    unsafe { libc::dlsym(libc::RTLD_DEFAULT, cname.as_ptr()) }
}
