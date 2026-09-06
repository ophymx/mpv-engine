//! Raw Windows plumbing for the exported-frame backend: a hidden WGL
//! context on an invisible window, GL framebuffers whose color storage
//! *is* a shared D3D11 texture (via `WGL_NV_DX_interop2`), and the
//! opengl32 symbol loader mpv renders through. opengl32, d3d11, dxgi,
//! user32, gdi32 and kernel32 are linked directly — no Rust dependency
//! enters the tree for this.
//!
//! Why D3D11 is the export currency here: Windows has no IOSurface and
//! no dmabuf, but every consumer API on the platform (D3D11, D3D12,
//! Vulkan via `VK_KHR_external_memory_win32`, and therefore wgpu) can
//! open a *shared NT handle* to a DXGI resource. `WGL_NV_DX_interop2`
//! is the one seam that lets a GL context render straight into such a
//! resource — it is what mpv's own `--gpu-context=dxinterop` and OBS
//! ride on — so the exportable framebuffer is a BGRA8
//! `D3D11_BIND_RENDER_TARGET` texture registered as a GL texture and
//! attached to an FBO.
//!
//! Two textures per buffer, not one. `wglDXRegisterObjectNV` refuses a
//! texture created with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` —
//! registration fails outright, verified on both the NVIDIA and Intel
//! ICDs — while `ID3D12Device::OpenSharedHandle`, and therefore wgpu's
//! DX12 import, accepts *only* NT handles. Those two requirements
//! cannot meet on one resource, so each pool buffer carries an interop
//! texture that GL renders into (legacy `D3D11_RESOURCE_MISC_SHARED`,
//! see [`D3D11_MISC_SHARED`]) plus an export texture that the shared NT
//! handle names, with a `CopyResource` between them in the publish
//! barrier. The copy is GPU-local on the render device — no readback —
//! but it does put the Windows tier one copy off the genuinely
//! zero-copy macOS and Linux paths.
//!
//! Thread discipline: everything that touches GL or the interop device —
//! context creation and currency, [`SurfaceBuffer::new`],
//! [`GlContext::begin_render`], [`GlContext::publish_barrier`],
//! [`SurfaceBuffer::delete_gl`] — runs only on the engine's export
//! render thread (the module's only GL caller), so GL's thread rules are
//! upheld by construction rather than by types. The D3D11 device is
//! free-threaded; its immediate context is not, so it lives behind a
//! mutex ([`D3d11::context`]) taken by [`SurfaceBuffer::copy_pixels`],
//! which any thread may call, by the publish barrier's `CopyResource`,
//! and by every `WGL_NV_DX_interop2` call — those drive the immediate
//! context inside the ICD, so being on the render thread does not
//! excuse them from the mutex. The shared NT handle is a kernel object, valid
//! from any thread or process — which is exactly why it is the export
//! currency (the IOSurface / dmabuf-fd analog).
//!
//! Buffer lifetime: each pool buffer owns two D3D11 textures, the export
//! texture's shared NT handle, and an interop registration plus GL
//! names. The export texture and handle travel across threads inside
//! [`ExportedFrame`](crate::ExportedFrame) and keep the memory alive on
//! their own — the buffer holds an `Arc` on the D3D11 device, so a frame
//! outliving the render thread stays readable; the GL and interop
//! handles are render-thread-only and die with the context. The same
//! split as macOS's retained IOSurface vs. GL names.

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::error::{Error, Result};

// ---- Win32 base types ----

type Handle = *mut c_void;
type Hwnd = *mut c_void;
type Hdc = *mut c_void;
type Hglrc = *mut c_void;
type Hmodule = *mut c_void;
type Bool32 = i32;
type Hresult = i32;

const NULL_HANDLE: Handle = std::ptr::null_mut();

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

/// `IID_IDXGIFactory1`.
const IID_IDXGI_FACTORY1: Guid = Guid {
    data1: 0x770a_ae78,
    data2: 0xf26f,
    data3: 0x4dba,
    data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
};
/// `IID_ID3D11Multithread`.
const IID_ID3D11_MULTITHREAD: Guid = Guid {
    data1: 0x9b7e_4e00,
    data2: 0x342c,
    data3: 0x4106,
    data4: [0xa1, 0x9f, 0x4f, 0x27, 0x04, 0xf6, 0x89, 0xf0],
};
/// `IID_IDXGIResource1`.
const IID_IDXGI_RESOURCE1: Guid = Guid {
    data1: 0x3096_1379,
    data2: 0x4609,
    data3: 0x4a41,
    data4: [0x99, 0x8e, 0x54, 0xfe, 0x56, 0x7e, 0xe0, 0xc1],
};

/// `DXGI_ERROR_NOT_FOUND` — `EnumAdapters1` past the last adapter.
const DXGI_ERROR_NOT_FOUND: Hresult = 0x887A_0002u32 as Hresult;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> Hmodule;
    fn GetModuleHandleA(name: *const c_char) -> Hmodule;
    fn GetProcAddress(module: Hmodule, name: *const c_char) -> *mut c_void;
    fn CloseHandle(object: Handle) -> Bool32;
}

// ---- The hidden window (user32 + gdi32) ----

const CS_OWNDC: u32 = 0x0020;
const WS_OVERLAPPED: u32 = 0;
const PFD_DRAW_TO_WINDOW: u32 = 0x0000_0004;
const PFD_SUPPORT_OPENGL: u32 = 0x0000_0020;
const PFD_DOUBLEBUFFER: u32 = 0x0000_0001;
const PFD_TYPE_RGBA: u8 = 0;
const PFD_MAIN_PLANE: u8 = 0;

#[repr(C)]
struct WndClassW {
    style: u32,
    wnd_proc: Option<unsafe extern "system" fn(Hwnd, u32, usize, isize) -> isize>,
    cls_extra: i32,
    wnd_extra: i32,
    instance: Hmodule,
    icon: Handle,
    cursor: Handle,
    background: Handle,
    menu_name: *const u16,
    class_name: *const u16,
}

#[repr(C)]
#[derive(Default)]
struct PixelFormatDescriptor {
    size: u16,
    version: u16,
    flags: u32,
    pixel_type: u8,
    color_bits: u8,
    red_bits: u8,
    red_shift: u8,
    green_bits: u8,
    green_shift: u8,
    blue_bits: u8,
    blue_shift: u8,
    alpha_bits: u8,
    alpha_shift: u8,
    accum_bits: u8,
    accum_red_bits: u8,
    accum_green_bits: u8,
    accum_blue_bits: u8,
    accum_alpha_bits: u8,
    depth_bits: u8,
    stencil_bits: u8,
    aux_buffers: u8,
    layer_type: u8,
    reserved: u8,
    layer_mask: u32,
    visible_mask: u32,
    damage_mask: u32,
}

#[link(name = "user32")]
unsafe extern "system" {
    fn RegisterClassW(class: *const WndClassW) -> u16;
    fn DefWindowProcW(hwnd: Hwnd, msg: u32, wparam: usize, lparam: isize) -> isize;
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: Hwnd,
        menu: Handle,
        instance: Hmodule,
        param: *mut c_void,
    ) -> Hwnd;
    fn DestroyWindow(hwnd: Hwnd) -> Bool32;
    fn GetDC(hwnd: Hwnd) -> Hdc;
    fn ReleaseDC(hwnd: Hwnd, hdc: Hdc) -> i32;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn ChoosePixelFormat(hdc: Hdc, pfd: *const PixelFormatDescriptor) -> i32;
    fn SetPixelFormat(hdc: Hdc, format: i32, pfd: *const PixelFormatDescriptor) -> Bool32;
}

// ---- WGL plus the GL 1.1 core opengl32 exports ----

const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_NEAREST: i32 = 0x2600;
const GL_RGBA8: u32 = 0x8058;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;

/// The FBO color format reported to mpv (`OpenGlFbo::internal_format`).
/// A hint only — the real storage layout is fixed by the D3D11 texture's
/// `DXGI_FORMAT_B8G8R8A8_UNORM`, which GL learns through the interop
/// registration.
pub(crate) const FBO_INTERNAL_FORMAT: i32 = GL_RGBA8 as i32;

/// `DXGI_FORMAT_B8G8R8A8_UNORM`: little-endian B,G,R,A bytes — the
/// layout Vulkan calls `B8G8R8A8_UNORM` and wgpu calls `Bgra8Unorm`,
/// and byte-identical to the macOS and Linux backends' output.
pub(crate) const DXGI_FORMAT_BGRA8_UNORM: u32 = 87;

#[link(name = "opengl32")]
unsafe extern "system" {
    fn wglCreateContext(hdc: Hdc) -> Hglrc;
    fn wglDeleteContext(ctx: Hglrc) -> Bool32;
    fn wglMakeCurrent(hdc: Hdc, ctx: Hglrc) -> Bool32;
    fn wglGetCurrentContext() -> Hglrc;
    fn wglGetProcAddress(name: *const c_char) -> *mut c_void;

    fn glGenTextures(n: i32, textures: *mut u32);
    fn glDeleteTextures(n: i32, textures: *const u32);
    fn glBindTexture(target: u32, texture: u32);
    fn glTexParameteri(target: u32, pname: u32, param: i32);
    fn glFinish();
}

// ---- WGL_NV_DX_interop2 and ARB_framebuffer_object (loaded, not
// linked: both are extensions, resolved against the live context) ----

/// `WGL_ACCESS_READ_WRITE_NV` — mpv both clears and blends into the
/// target, so the GL side must be able to read it back.
const WGL_ACCESS_READ_WRITE_NV: u32 = 0x0000_0001;

type PfnDxOpenDevice = unsafe extern "system" fn(*mut c_void) -> Handle;
type PfnDxCloseDevice = unsafe extern "system" fn(Handle) -> Bool32;
type PfnDxRegisterObject = unsafe extern "system" fn(Handle, *mut c_void, u32, u32, u32) -> Handle;
type PfnDxUnregisterObject = unsafe extern "system" fn(Handle, Handle) -> Bool32;
type PfnDxLockObjects = unsafe extern "system" fn(Handle, i32, *mut Handle) -> Bool32;

#[allow(non_snake_case)]
struct GlFns {
    GenFramebuffers: unsafe extern "system" fn(i32, *mut u32),
    DeleteFramebuffers: unsafe extern "system" fn(i32, *const u32),
    BindFramebuffer: unsafe extern "system" fn(u32, u32),
    FramebufferTexture2D: unsafe extern "system" fn(u32, u32, u32, u32, i32),
    CheckFramebufferStatus: unsafe extern "system" fn(u32) -> u32,
    DXOpenDevice: PfnDxOpenDevice,
    DXCloseDevice: PfnDxCloseDevice,
    DXRegisterObject: PfnDxRegisterObject,
    DXUnregisterObject: PfnDxUnregisterObject,
    DXLockObjects: PfnDxLockObjects,
    DXUnlockObjects: PfnDxLockObjects,
}

/// Load one entry point through [`gl_proc_address`] as the given fn
/// type, erroring on NULL.
macro_rules! load_fn {
    ($name:literal as $ty:ty) => {{
        let ptr = gl_proc_address($name);
        if ptr.is_null() {
            return Err(Error::ExportSetup(
                concat!("required GL/WGL entry point missing: ", $name).into(),
            ));
        }
        // SAFETY: resolved from the live GL implementation under the
        // name whose documented signature `$ty` spells.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
    }};
}

// ---- D3D11 / DXGI, hand-rolled COM ----

const D3D_DRIVER_TYPE_UNKNOWN: u32 = 0;
const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
const D3D11_SDK_VERSION: u32 = 7;
/// `D3D11_CREATE_DEVICE_BGRA_SUPPORT` — required for BGRA render
/// targets on some feature levels, and free everywhere else.
const D3D11_CREATE_DEVICE_BGRA_SUPPORT: u32 = 0x20;

const D3D11_USAGE_DEFAULT: u32 = 0;
const D3D11_USAGE_STAGING: u32 = 3;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
const D3D11_CPU_ACCESS_READ: u32 = 0x0002_0000;
/// `D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`.
/// Deliberately *not* `_KEYEDMUTEX`: a keyed mutex would have to be
/// acquired by the consumer, and neither wgpu nor a plain
/// `OpenSharedHandle` importer knows to do that. Cross-API ordering is
/// carried by the publish barrier instead — the same discipline as the
/// Linux backend's `glFinish` (see [`GlContext::publish_barrier`]).
const D3D11_MISC_SHARED_NTHANDLE: u32 = 0x0002 | 0x0800;
/// `D3D11_RESOURCE_MISC_SHARED` — the *legacy* shared flag, and the only
/// one `wglDXRegisterObjectNV` will register. Adding
/// `_NTHANDLE` to it makes registration fail on every ICD tested, which
/// is why the interop texture and the export texture are separate
/// resources (see the module docs).
const D3D11_MISC_SHARED: u32 = 0x0002;
const D3D11_MAP_READ: u32 = 1;
/// `GENERIC_ALL` — the access mask `CreateSharedHandle` wants for a
/// resource the consumer may both read and (in principle) write.
const GENERIC_ALL: u32 = 0x1000_0000;

#[repr(C)]
struct Texture2dDesc {
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
}

#[repr(C)]
struct MappedSubresource {
    data: *mut c_void,
    row_pitch: u32,
    depth_pitch: u32,
}

/// Vtable prefixes: only the methods this module calls are typed, and
/// everything before them is an opaque slot — so the layout below spells
/// the real COM indices, and nothing past the last named method is ever
/// dereferenced.
#[repr(C)]
struct IUnknownVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct ID3D11DeviceVtbl {
    _unknown: IUnknownVtbl,
    _create_buffer: *const c_void,
    _create_texture_1d: *const c_void,
    /// 5.
    create_texture_2d: unsafe extern "system" fn(
        *mut c_void,
        *const Texture2dDesc,
        *const c_void,
        *mut *mut c_void,
    ) -> Hresult,
}

#[repr(C)]
struct ID3D11DeviceContextVtbl {
    _unknown: IUnknownVtbl,
    /// `ID3D11DeviceChild` (3..=6).
    _device_child: [*const c_void; 4],
    /// 7..=13.
    _before_map: [*const c_void; 7],
    /// 14.
    map: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        u32,
        u32,
        *mut MappedSubresource,
    ) -> Hresult,
    /// 15.
    unmap: unsafe extern "system" fn(*mut c_void, *mut c_void, u32),
    /// 16..=46.
    _before_copy_resource: [*const c_void; 31],
    /// 47.
    copy_resource: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void),
    /// 48..=110.
    _before_flush: [*const c_void; 63],
    /// 111.
    flush: unsafe extern "system" fn(*mut c_void),
}

#[repr(C)]
struct ID3D11MultithreadVtbl {
    _unknown: IUnknownVtbl,
    /// 3, 4.
    _enter_leave: [*const c_void; 2],
    /// 5.
    set_multithread_protected: unsafe extern "system" fn(*mut c_void, Bool32) -> Bool32,
}

#[repr(C)]
struct IDXGIResource1Vtbl {
    _unknown: IUnknownVtbl,
    /// `IDXGIObject` (3..=6).
    _object: [*const c_void; 4],
    /// `IDXGIDeviceSubObject` (7).
    _device_subobject: *const c_void,
    /// `IDXGIResource` (8..=11).
    _resource: [*const c_void; 4],
    /// 12.
    _create_subresource_surface: *const c_void,
    /// 13.
    create_shared_handle: unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        u32,
        *const u16,
        *mut Handle,
    ) -> Hresult,
}

#[repr(C)]
struct IDXGIFactory1Vtbl {
    _unknown: IUnknownVtbl,
    /// `IDXGIObject` (3..=6).
    _object: [*const c_void; 4],
    /// `IDXGIFactory` (7..=11).
    _factory: [*const c_void; 5],
    /// 12.
    enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> Hresult,
}

/// Borrow a COM object's vtable. An interface pointer's first word *is*
/// the vtable pointer, by the ABI's definition.
unsafe fn vtbl<'a, V>(obj: *mut c_void) -> &'a V {
    unsafe { &**obj.cast::<*const V>() }
}

unsafe fn com_release(obj: *mut c_void) {
    if !obj.is_null() {
        unsafe { (vtbl::<IUnknownVtbl>(obj).release)(obj) };
    }
}

#[link(name = "d3d11")]
unsafe extern "system" {
    fn D3D11CreateDevice(
        adapter: *mut c_void,
        driver_type: u32,
        software: Hmodule,
        flags: u32,
        feature_levels: *const u32,
        num_feature_levels: u32,
        sdk_version: u32,
        device: *mut *mut c_void,
        feature_level: *mut u32,
        immediate_context: *mut *mut c_void,
    ) -> Hresult;
}

#[link(name = "dxgi")]
unsafe extern "system" {
    fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> Hresult;
}

fn hresult_error(what: &str, hr: Hresult) -> Error {
    Error::ExportSetup(format!("{what}: HRESULT {:#010x}", hr as u32))
}

// ---- The D3D11 device shared by the context and every buffer ----

/// The D3D11 device the exportable textures are allocated on, plus its
/// immediate context (for CPU readback only). Held through an `Arc` by
/// the [`GlContext`] *and* by every [`SurfaceBuffer`], so a frame the
/// shell is still holding stays readable after the render thread is
/// gone — the Windows counterpart to a retained IOSurface, or to a
/// dmabuf fd outliving its EGL display.
struct D3d11 {
    device: *mut c_void,
    /// D3D11 devices are free-threaded; immediate contexts are not.
    /// This mutex is the module's one serialization point for the
    /// immediate context, and it covers more than the calls that name it
    /// directly: `WGL_NV_DX_interop2`'s open/register/lock/unlock/
    /// unregister/close all drive the immediate context inside the ICD,
    /// so they take it too. Leaving the interop calls out is not a
    /// theoretical hazard — the render thread hangs inside
    /// `wglDXLockObjectsNV` when the shell calls
    /// [`SurfaceBuffer::copy_pixels`] at the same moment, which an
    /// acquire-and-inspect consumer does on every frame.
    context: Mutex<*mut c_void>,
}

// SAFETY: the device is free-threaded by D3D11's own contract, and the
// immediate context is only ever touched under `context`'s mutex.
unsafe impl Send for D3d11 {}
unsafe impl Sync for D3d11 {}

impl D3d11 {
    /// Create a device on `adapter` (null = whatever
    /// `D3D_DRIVER_TYPE_HARDWARE` picks).
    fn create(adapter: *mut c_void) -> Result<Self> {
        let driver_type = if adapter.is_null() {
            D3D_DRIVER_TYPE_HARDWARE
        } else {
            // An explicit adapter obligates UNKNOWN — D3D11 returns
            // E_INVALIDARG otherwise.
            D3D_DRIVER_TYPE_UNKNOWN
        };
        let mut device: *mut c_void = std::ptr::null_mut();
        let mut context: *mut c_void = std::ptr::null_mut();
        let hr = unsafe {
            D3D11CreateDevice(
                adapter,
                driver_type,
                std::ptr::null_mut(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                std::ptr::null(),
                0,
                D3D11_SDK_VERSION,
                &mut device,
                std::ptr::null_mut(),
                &mut context,
            )
        };
        if hr < 0 || device.is_null() || context.is_null() {
            unsafe {
                com_release(context);
                com_release(device);
            }
            return Err(hresult_error("D3D11CreateDevice", hr));
        }
        // Belt and braces with `context`'s mutex below. That mutex covers
        // every immediate-context call *this module* makes, including the
        // interop entry points; multithread protection additionally covers
        // the ones the ICD makes behind our back — `wglDXLockObjectsNV`
        // and friends drive the immediate context internally, and so does
        // the D3D11-on-D3D12/DXGI plumbing under some drivers. Failing to
        // acquire the interface is not fatal: our own mutex is what the
        // correctness argument rests on.
        unsafe {
            let mut multithread: *mut c_void = std::ptr::null_mut();
            let hr = (vtbl::<IUnknownVtbl>(context).query_interface)(
                context,
                &IID_ID3D11_MULTITHREAD,
                &mut multithread,
            );
            if hr >= 0 && !multithread.is_null() {
                (vtbl::<ID3D11MultithreadVtbl>(multithread).set_multithread_protected)(
                    multithread,
                    1,
                );
                com_release(multithread);
            } else {
                tracing::debug!("export: ID3D11Multithread unavailable (HRESULT {hr:#010x})");
            }
        }
        Ok(Self {
            device,
            context: Mutex::new(context),
        })
    }
}

impl Drop for D3d11 {
    fn drop(&mut self) {
        unsafe {
            com_release(*self.context.get_mut());
            com_release(self.device);
        }
    }
}

/// Every DXGI adapter, in the factory's own order. Empty when the
/// factory itself is unavailable — [`GlContext::new`] then falls back to
/// the one "let D3D11 pick" attempt. Callers must release each adapter.
fn enumerate_adapters() -> Vec<*mut c_void> {
    let mut factory: *mut c_void = std::ptr::null_mut();
    let hr = unsafe { CreateDXGIFactory1(&IID_IDXGI_FACTORY1, &mut factory) };
    if hr < 0 || factory.is_null() {
        return Vec::new();
    }
    let mut adapters = Vec::new();
    let mut index = 0u32;
    loop {
        let mut adapter: *mut c_void = std::ptr::null_mut();
        let hr = unsafe {
            (vtbl::<IDXGIFactory1Vtbl>(factory).enum_adapters1)(factory, index, &mut adapter)
        };
        if hr == DXGI_ERROR_NOT_FOUND || hr < 0 || adapter.is_null() {
            break;
        }
        adapters.push(adapter);
        index += 1;
    }
    unsafe { com_release(factory) };
    adapters
}

// ---- RAII pieces so `GlContext::new`'s error paths unwind cleanly ----

/// The invisible window whose DC carries the pixel format. Never shown
/// and never pumped: WGL wants a DC with a pixel format and nothing
/// more, because every rendering target here is an FBO.
struct HiddenWindow {
    hwnd: Hwnd,
    hdc: Hdc,
}

impl HiddenWindow {
    fn new() -> Result<Self> {
        // "mpv-engine-export", UTF-16 and NUL-terminated.
        const CLASS_NAME: &[u16] = &[
            0x6d, 0x70, 0x76, 0x2d, 0x65, 0x6e, 0x67, 0x69, 0x6e, 0x65, 0x2d, 0x65, 0x78, 0x70,
            0x6f, 0x72, 0x74, 0x00,
        ];
        let instance = unsafe { GetModuleHandleW(std::ptr::null()) };
        // The class is process-wide and outlives any one context, so
        // register it exactly once; a repeat registration would fail and
        // `CreateWindowExW` below is the real check anyway.
        static REGISTERED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        REGISTERED.get_or_init(|| {
            let class = WndClassW {
                style: CS_OWNDC,
                wnd_proc: Some(DefWindowProcW),
                cls_extra: 0,
                wnd_extra: 0,
                instance,
                icon: NULL_HANDLE,
                cursor: NULL_HANDLE,
                background: NULL_HANDLE,
                menu_name: std::ptr::null(),
                class_name: CLASS_NAME.as_ptr(),
            };
            unsafe { RegisterClassW(&class) };
        });
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                CLASS_NAME.as_ptr(),
                CLASS_NAME.as_ptr(),
                WS_OVERLAPPED,
                0,
                0,
                1,
                1,
                std::ptr::null_mut(),
                NULL_HANDLE,
                instance,
                std::ptr::null_mut(),
            )
        };
        if hwnd.is_null() {
            return Err(Error::ExportSetup(
                "CreateWindowExW failed for the hidden GL window (a session with no window \
                 station — a service, or SSH without an interactive desktop?)"
                    .into(),
            ));
        }
        let hdc = unsafe { GetDC(hwnd) };
        if hdc.is_null() {
            unsafe { DestroyWindow(hwnd) };
            return Err(Error::ExportSetup(
                "GetDC failed for the hidden window".into(),
            ));
        }
        Ok(Self { hwnd, hdc })
    }

    /// Pick and set the DC's pixel format. Any RGBA format does — the
    /// default framebuffer is never drawn to — so this is a plain
    /// `ChoosePixelFormat` on a minimal descriptor rather than the
    /// `WGL_ARB_pixel_format` two-context dance.
    fn set_pixel_format(&self) -> Result<()> {
        let pfd = PixelFormatDescriptor {
            size: std::mem::size_of::<PixelFormatDescriptor>() as u16,
            version: 1,
            flags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            pixel_type: PFD_TYPE_RGBA,
            color_bits: 32,
            alpha_bits: 8,
            depth_bits: 24,
            stencil_bits: 8,
            layer_type: PFD_MAIN_PLANE,
            ..Default::default()
        };
        let format = unsafe { ChoosePixelFormat(self.hdc, &pfd) };
        if format == 0 {
            return Err(Error::ExportSetup(
                "ChoosePixelFormat found no OpenGL-capable format".into(),
            ));
        }
        if unsafe { SetPixelFormat(self.hdc, format, &pfd) } == 0 {
            return Err(Error::ExportSetup("SetPixelFormat failed".into()));
        }
        Ok(())
    }
}

impl Drop for HiddenWindow {
    fn drop(&mut self) {
        unsafe {
            ReleaseDC(self.hwnd, self.hdc);
            DestroyWindow(self.hwnd);
        }
    }
}

/// Half-built WGL context, so [`GlContext::new`]'s error paths unwind;
/// forgotten once it moves into the finished context.
struct WglContext {
    hglrc: Hglrc,
}

impl Drop for WglContext {
    fn drop(&mut self) {
        unsafe {
            if wglGetCurrentContext() == self.hglrc {
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
            }
            wglDeleteContext(self.hglrc);
        }
    }
}

/// The hidden GL context the export render thread owns: a WGL context on
/// an invisible window's DC, tied to a D3D11 device through
/// `WGL_NV_DX_interop2`. Created, used, and dropped on that one thread.
pub(crate) struct GlContext {
    hglrc: Hglrc,
    /// The interop device tying the GL context to `d3d` — the handle
    /// every `wglDX*ObjectNV` call goes through.
    dx_device: Handle,
    fns: GlFns,
    // Teardown order = declaration order once `Drop`'s explicit part has
    // closed the interop device and the GL context: the D3D11 device
    // (released only when no frame still holds it), then the window
    // whose DC that context rendered on.
    d3d: Arc<D3d11>,
    window: HiddenWindow,
}

impl GlContext {
    /// Size of the throwaway buffer a candidate adapter is proved with.
    /// Small enough to cost nothing; the registration is what is under
    /// test, not the dimensions.
    const PROBE_SIZE: u32 = 64;

    /// Bring the backend up on the first adapter that actually *works*.
    /// A GL context and a D3D11 device that each come up fine can still
    /// refuse to interop when they landed on different GPUs — the
    /// ordinary case on a hybrid laptop, where the GL ICD picks the
    /// discrete chip while DXGI's adapter 0 is the integrated one — so
    /// every failure falls through to the next adapter (the same
    /// device-selection discipline as the Linux backend's render-node
    /// ladder) rather than failing the attach.
    ///
    /// "Works" has to mean a *registered buffer*, not an opened interop
    /// device. `wglDXOpenDeviceNV` returns a live handle for a
    /// mismatched GL/D3D pair on at least some ICDs, and the rejection
    /// only lands later, on the first `wglDXRegisterObjectNV`. Accepting
    /// an adapter on the open alone therefore produces the worst
    /// possible outcome: an attach that reports success, and a render
    /// loop that retries the failing allocation forever without ever
    /// publishing a frame. Each candidate is proved with a throwaway
    /// buffer here instead, so a bad adapter is a fall-through and total
    /// failure is an honest `ExportSetup` error the shell can see.
    pub(crate) fn new() -> Result<Self> {
        let adapters = enumerate_adapters();
        // The null entry is the "let D3D11 pick" fallback, so a box
        // whose DXGI factory refused to enumerate still gets one try.
        let candidates: Vec<*mut c_void> = if adapters.is_empty() {
            vec![std::ptr::null_mut()]
        } else {
            adapters.clone()
        };
        let mut last_error = None;
        let mut opened = None;
        for adapter in candidates {
            match Self::try_adapter(adapter) {
                Ok(context) => {
                    opened = Some(context);
                    break;
                }
                Err(e) => {
                    tracing::debug!("export: adapter unusable for GL interop: {e}");
                    last_error = Some(e);
                }
            }
        }
        for adapter in adapters {
            unsafe { com_release(adapter) };
        }
        opened.ok_or_else(|| {
            last_error.unwrap_or_else(|| {
                Error::ExportSetup("no D3D11 adapter could be opened for GL interop".into())
            })
        })
    }

    /// One rung of the ladder: the whole backend on one adapter, proved
    /// end to end. The window and GL context are rebuilt per candidate
    /// rather than shared across the loop so that a rejected rung tears
    /// down through the ordinary `Drop` — which closes the interop
    /// device while its GL context is still current and its D3D11 device
    /// still alive, the only order the extension allows. Both are cheap,
    /// and this runs once per attach.
    fn try_adapter(adapter: *mut c_void) -> Result<Self> {
        let window = HiddenWindow::new()?;
        window.set_pixel_format()?;
        // A plain `wglCreateContext` yields the driver's best
        // compatibility profile (4.6 on every current ICD), which is
        // what mpv is happiest on — the `wglCreateContextAttribsARB`
        // ladder would buy nothing the CGL profile ladder buys on macOS.
        // Creating it first is not optional either way:
        // `wglGetProcAddress` resolves nothing without a current
        // context.
        let hglrc = unsafe { wglCreateContext(window.hdc) };
        if hglrc.is_null() {
            return Err(Error::ExportSetup(
                "wglCreateContext failed (no OpenGL ICD? the Microsoft software rasterizer is \
                 GL 1.1 and cannot drive mpv)"
                    .into(),
            ));
        }
        let context = WglContext { hglrc };
        if unsafe { wglMakeCurrent(window.hdc, context.hglrc) } == 0 {
            return Err(Error::ExportSetup("wglMakeCurrent failed".into()));
        }
        let fns = GlFns {
            GenFramebuffers: load_fn!(
                "glGenFramebuffers" as unsafe extern "system" fn(i32, *mut u32)
            ),
            DeleteFramebuffers: load_fn!(
                "glDeleteFramebuffers" as unsafe extern "system" fn(i32, *const u32)
            ),
            BindFramebuffer: load_fn!("glBindFramebuffer" as unsafe extern "system" fn(u32, u32)),
            FramebufferTexture2D: load_fn!(
                "glFramebufferTexture2D" as unsafe extern "system" fn(u32, u32, u32, u32, i32)
            ),
            CheckFramebufferStatus: load_fn!(
                "glCheckFramebufferStatus" as unsafe extern "system" fn(u32) -> u32
            ),
            DXOpenDevice: load_fn!("wglDXOpenDeviceNV" as PfnDxOpenDevice),
            DXCloseDevice: load_fn!("wglDXCloseDeviceNV" as PfnDxCloseDevice),
            DXRegisterObject: load_fn!("wglDXRegisterObjectNV" as PfnDxRegisterObject),
            DXUnregisterObject: load_fn!("wglDXUnregisterObjectNV" as PfnDxUnregisterObject),
            DXLockObjects: load_fn!("wglDXLockObjectsNV" as PfnDxLockObjects),
            DXUnlockObjects: load_fn!("wglDXUnlockObjectsNV" as PfnDxLockObjects),
        };

        let d3d = D3d11::create(adapter)?;
        let dx_device = unsafe { (fns.DXOpenDevice)(d3d.device) };
        if dx_device.is_null() {
            return Err(Error::ExportSetup(
                "wglDXOpenDeviceNV refused the D3D11 device (GL and D3D on different \
                 GPUs, or no WGL_NV_DX_interop2 support)"
                    .into(),
            ));
        }
        let hglrc = context.hglrc;
        std::mem::forget(context);
        let gl = Self {
            hglrc,
            dx_device,
            fns,
            d3d: Arc::new(d3d),
            window,
        };
        // The proof. Dropping `gl` on failure runs the full teardown, so
        // the next rung starts from a clean slate.
        let mut probe = SurfaceBuffer::new(&gl, Self::PROBE_SIZE, Self::PROBE_SIZE)?;
        probe.delete_gl(&gl);
        Ok(gl)
    }

    pub(crate) fn make_current(&self) -> Result<()> {
        if unsafe { wglMakeCurrent(self.window.hdc, self.hglrc) } == 0 {
            return Err(Error::ExportSetup("wglMakeCurrent failed".into()));
        }
        Ok(())
    }

    /// Hand the buffer's storage to GL before mpv renders into it. Under
    /// `WGL_NV_DX_interop2` a registered object's GL texture has usable
    /// storage only while *locked*, and D3D — hence anything that opened
    /// the shared handle — may only touch it while *unlocked*. Locking
    /// here and unlocking in
    /// [`publish_barrier`](Self::publish_barrier) is what makes that
    /// handover per-frame instead of a race.
    ///
    /// A no-op on macOS and Linux, whose exportable memory needs no
    /// handover.
    pub(crate) fn begin_render(&self, buffer: &SurfaceBuffer) {
        buffer.lock_for_gl(self);
    }

    /// Publish barrier before the shared texture is consumed by another
    /// API. Two things happen: `glFinish` retires mpv's GPU work — a
    /// shared NT handle carries no cross-API ordering guarantee, the
    /// consumer's D3D12/Vulkan device is a *different* device, and this
    /// backend deliberately allocates without a keyed mutex so wgpu can
    /// import the texture at all (same reasoning as the Linux dmabuf
    /// path) — and the interop unlock hands the texture back to D3D so
    /// the copy that follows is legal. [`SurfaceBuffer::copy_to_shared`]
    /// then fills the export texture and flushes, which is what makes
    /// the handle's openers see finished pixels.
    pub(crate) fn publish_barrier(&self, buffer: &SurfaceBuffer) {
        unsafe { glFinish() };
        buffer.unlock_from_gl(self);
        buffer.copy_to_shared();
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            // The interop device closes first: it is the only thing here
            // that needs both the GL context current and the D3D11
            // device alive. Under the immediate-context mutex, like
            // every other interop call: a frame the shell still holds
            // can be running `copy_pixels` on another thread right now.
            let ctx = self.d3d.context.lock();
            (self.fns.DXCloseDevice)(self.dx_device);
            drop(ctx);
            if wglGetCurrentContext() == self.hglrc {
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
            }
            wglDeleteContext(self.hglrc);
        }
        // RAII fields finish the teardown in declaration order.
    }
}

/// One pool entry: a pair of BGRA8 `D3D11_BIND_RENDER_TARGET` textures —
/// the interop one registered through `WGL_NV_DX_interop2` as a GL
/// texture and attached to a framebuffer mpv renders into, and the
/// export one carrying the shared NT handle, filled from it by the
/// publish barrier (the module docs explain why one texture cannot do
/// both jobs). The texture
/// and handle travel across threads inside
/// [`ExportedFrame`](crate::ExportedFrame); the GL name and the interop
/// object are handles only the render thread ever dereferences.
pub(crate) struct SurfaceBuffer {
    /// Held by every buffer, not just by the context: an outstanding
    /// frame must stay readable after detach.
    d3d: Arc<D3d11>,
    /// The exported half: NT-handle-shareable, never registered with GL,
    /// written only by the publish barrier's `CopyResource`.
    texture: *mut c_void,
    /// The GL half: legacy-shared so `wglDXRegisterObjectNV` accepts it,
    /// never handed to a consumer.
    interop_texture: *mut c_void,
    handle: Handle,
    dx_object: Handle,
    gl_texture: u32,
    fbo: u32,
    /// Whether the interop object is currently locked for GL. Render
    /// thread only; atomic so the buffer stays `Sync` without a mutex.
    locked: AtomicBool,
    width: u32,
    height: u32,
}

// SAFETY: the D3D11 texture is free-threaded (read only through the
// mutex-guarded immediate context) and the shared handle is a kernel
// object. `dx_object` and the GL name are plain handles here; every call
// against them (creation, locking, deletion, mpv's renders) happens on
// the render thread — see the module docs' thread discipline. `Drop`
// touches only the texture's refcount and the handle.
unsafe impl Send for SurfaceBuffer {}
unsafe impl Sync for SurfaceBuffer {}

impl SurfaceBuffer {
    /// Create a `width`×`height` buffer. Render thread only, GL context
    /// current.
    pub(crate) fn new(gl: &GlContext, width: u32, height: u32) -> Result<Self> {
        let interop_texture =
            create_shared_texture(gl.d3d.device, width, height, D3D11_MISC_SHARED)?;
        let texture =
            match create_shared_texture(gl.d3d.device, width, height, D3D11_MISC_SHARED_NTHANDLE) {
                Ok(texture) => texture,
                Err(e) => {
                    unsafe { com_release(interop_texture) };
                    return Err(e);
                }
            };
        let handle = match create_shared_handle(texture) {
            Ok(handle) => handle,
            Err(e) => {
                unsafe {
                    com_release(texture);
                    com_release(interop_texture);
                }
                return Err(e);
            }
        };
        // From here on the value's own `Drop` covers the D3D side, and
        // every early return goes through `delete_gl` for the GL side.
        let mut buffer = Self {
            d3d: Arc::clone(&gl.d3d),
            texture,
            interop_texture,
            handle,
            dx_object: NULL_HANDLE,
            gl_texture: 0,
            fbo: 0,
            locked: AtomicBool::new(false),
            width,
            height,
        };
        let f = &gl.fns;
        unsafe {
            glGenTextures(1, &mut buffer.gl_texture);
            // Registration touches the immediate context too (see
            // `lock_for_gl`); the guard is scoped so the `lock_for_gl`
            // below — the mutex is not reentrant — takes it afresh.
            let _ctx = gl.d3d.context.lock();
            buffer.dx_object = (f.DXRegisterObject)(
                gl.dx_device,
                interop_texture,
                buffer.gl_texture,
                GL_TEXTURE_2D,
                WGL_ACCESS_READ_WRITE_NV,
            );
        }
        if buffer.dx_object.is_null() {
            buffer.delete_gl(gl);
            return Err(Error::ExportSetup(format!(
                "wglDXRegisterObjectNV failed for a {width}x{height} BGRA8 render target"
            )));
        }
        // The registered texture only has storage while locked, so the
        // framebuffer is built and validated inside a lock.
        buffer.lock_for_gl(gl);
        let status = unsafe {
            (f.GenFramebuffers)(1, &mut buffer.fbo);
            (f.BindFramebuffer)(GL_FRAMEBUFFER, buffer.fbo);
            (f.FramebufferTexture2D)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                buffer.gl_texture,
                0,
            );
            let status = (f.CheckFramebufferStatus)(GL_FRAMEBUFFER);
            (f.BindFramebuffer)(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, buffer.gl_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);
            status
        };
        buffer.unlock_from_gl(gl);
        if status != GL_FRAMEBUFFER_COMPLETE {
            buffer.delete_gl(gl);
            return Err(Error::ExportSetup(format!(
                "shared-texture framebuffer incomplete (status {status:#x})"
            )));
        }
        Ok(buffer)
    }

    /// Take the interop object for GL, idempotently — re-locking a
    /// locked object is an error in the extension, and the render path
    /// pairs lock/unlock across a *fallible* mpv render.
    fn lock_for_gl(&self, gl: &GlContext) {
        if self.dx_object.is_null() || self.locked.swap(true, Ordering::SeqCst) {
            return;
        }
        let mut object = self.dx_object;
        // Under the immediate-context mutex: the ICD implements the lock
        // by driving the D3D11 immediate context, which `copy_pixels`
        // may be using from the shell's thread at this very moment (it
        // runs on every acquired frame). Racing the two hangs the render
        // thread inside the driver — see `D3d11::context`.
        let _ctx = self.d3d.context.lock();
        if unsafe { (gl.fns.DXLockObjects)(gl.dx_device, 1, &mut object) } == 0 {
            tracing::warn!("export: wglDXLockObjectsNV failed; this frame may be stale");
        }
    }

    /// Hand the interop object back to D3D, idempotently.
    fn unlock_from_gl(&self, gl: &GlContext) {
        if self.dx_object.is_null() || !self.locked.swap(false, Ordering::SeqCst) {
            return;
        }
        let mut object = self.dx_object;
        // Same immediate-context serialization as `lock_for_gl`.
        let _ctx = self.d3d.context.lock();
        if unsafe { (gl.fns.DXUnlockObjects)(gl.dx_device, 1, &mut object) } == 0 {
            tracing::warn!("export: wglDXUnlockObjectsNV failed; this frame may be torn");
        }
    }

    /// Fill the export texture from the one GL just rendered into.
    /// Called by the publish barrier *after* the interop unlock, because
    /// D3D may only touch a registered resource while it is unlocked.
    ///
    /// The `Flush` is not optional: the consumer opens the shared handle
    /// on a *different* device, and with no keyed mutex in play nothing
    /// else forces this copy out of the immediate context's queue before
    /// that device reads the texture.
    fn copy_to_shared(&self) {
        let ctx = self.d3d.context.lock();
        unsafe {
            let v = vtbl::<ID3D11DeviceContextVtbl>(*ctx);
            (v.copy_resource)(*ctx, self.texture, self.interop_texture);
            (v.flush)(*ctx);
        }
    }

    pub(crate) fn fbo(&self) -> u32 {
        self.fbo
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn shared_handle(&self) -> Handle {
        self.handle
    }

    /// Delete the GL texture/framebuffer names and the interop
    /// registration. Render thread only, GL context current; the D3D11
    /// texture and its shared handle are released by `Drop` (any
    /// thread). Buffers retired anywhere else simply skip this — their
    /// GL name and interop object die with the render thread's context.
    pub(crate) fn delete_gl(&mut self, gl: &GlContext) {
        // Unregistering a locked object is invalid, and a buffer can
        // reach here locked if mpv's render errored out mid-frame.
        self.unlock_from_gl(gl);
        let f = &gl.fns;
        unsafe {
            if self.fbo != 0 {
                (f.DeleteFramebuffers)(1, &self.fbo);
                self.fbo = 0;
            }
            if !self.dx_object.is_null() {
                let ctx = self.d3d.context.lock();
                (f.DXUnregisterObject)(gl.dx_device, self.dx_object);
                drop(ctx);
                self.dx_object = NULL_HANDLE;
            }
            if self.gl_texture != 0 {
                glDeleteTextures(1, &self.gl_texture);
                self.gl_texture = 0;
            }
        }
    }

    /// Copy the texture's pixels out as tightly packed BGRA rows
    /// (row 0 = top). Any thread: this goes through a staging texture on
    /// the D3D11 immediate context — no GL and no interop — so it works
    /// even after the render context is gone (the buffer's `Arc<D3d11>`
    /// keeps the device alive). Returns zeroed pixels if the staging
    /// copy or the map fails, mirroring the other platforms'
    /// lock-failure behavior.
    pub(crate) fn copy_pixels(&self) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = vec![0u8; w * h * 4];
        if w == 0 || h == 0 {
            return out;
        }
        let desc = Texture2dDesc {
            width: self.width,
            height: self.height,
            mip_levels: 1,
            array_size: 1,
            format: DXGI_FORMAT_BGRA8_UNORM,
            sample_count: 1,
            sample_quality: 0,
            usage: D3D11_USAGE_STAGING,
            bind_flags: 0,
            cpu_access_flags: D3D11_CPU_ACCESS_READ,
            misc_flags: 0,
        };
        let device = self.d3d.device;
        let mut staging: *mut c_void = std::ptr::null_mut();
        let hr = unsafe {
            (vtbl::<ID3D11DeviceVtbl>(device).create_texture_2d)(
                device,
                &desc,
                std::ptr::null(),
                &mut staging,
            )
        };
        if hr < 0 || staging.is_null() {
            return out;
        }
        let ctx = self.d3d.context.lock();
        unsafe {
            let v = vtbl::<ID3D11DeviceContextVtbl>(*ctx);
            (v.copy_resource)(*ctx, staging, self.texture);
            let mut mapped = MappedSubresource {
                data: std::ptr::null_mut(),
                row_pitch: 0,
                depth_pitch: 0,
            };
            if (v.map)(*ctx, staging, 0, D3D11_MAP_READ, 0, &mut mapped) >= 0
                && !mapped.data.is_null()
            {
                let base = mapped.data.cast::<u8>();
                let pitch = mapped.row_pitch as usize;
                for row in 0..h {
                    std::ptr::copy_nonoverlapping(
                        base.add(row * pitch),
                        out.as_mut_ptr().add(row * w * 4),
                        w * 4,
                    );
                }
                (v.unmap)(*ctx, staging, 0);
            }
            com_release(staging);
        }
        out
    }
}

impl Drop for SurfaceBuffer {
    fn drop(&mut self) {
        // Only the D3D side needs an explicit release here; GL names and
        // the interop registration are handled per `delete_gl`'s
        // contract. Closing our handle does not free the memory while a
        // consumer's opened copy still references it.
        unsafe {
            if !self.handle.is_null() {
                CloseHandle(self.handle);
            }
            com_release(self.texture);
            com_release(self.interop_texture);
        }
    }
}

/// Allocate one of a buffer's two textures: BGRA8 (the one format every
/// Windows consumer — D3D11, D3D12, Vulkan, wgpu `Bgra8Unorm` — takes
/// without conversion), renderable and shader-readable. `misc_flags`
/// picks which half it is: [`D3D11_MISC_SHARED`] for the texture GL
/// registers, [`D3D11_MISC_SHARED_NTHANDLE`] for the one the export
/// handle names. Neither takes a keyed mutex.
fn create_shared_texture(
    device: *mut c_void,
    width: u32,
    height: u32,
    misc_flags: u32,
) -> Result<*mut c_void> {
    let desc = Texture2dDesc {
        width,
        height,
        mip_levels: 1,
        array_size: 1,
        format: DXGI_FORMAT_BGRA8_UNORM,
        sample_count: 1,
        sample_quality: 0,
        usage: D3D11_USAGE_DEFAULT,
        bind_flags: D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        cpu_access_flags: 0,
        misc_flags,
    };
    let mut texture: *mut c_void = std::ptr::null_mut();
    let hr = unsafe {
        (vtbl::<ID3D11DeviceVtbl>(device).create_texture_2d)(
            device,
            &desc,
            std::ptr::null(),
            &mut texture,
        )
    };
    if hr < 0 || texture.is_null() {
        return Err(hresult_error(
            &format!("CreateTexture2D for a shared {width}x{height} BGRA8 render target"),
            hr,
        ));
    }
    Ok(texture)
}

/// Export the texture's shared NT handle — the handle
/// [`ExportedFrame::shared_handle`](crate::ExportedFrame::shared_handle)
/// gives the shell.
fn create_shared_handle(texture: *mut c_void) -> Result<Handle> {
    let mut resource: *mut c_void = std::ptr::null_mut();
    let hr = unsafe {
        (vtbl::<IUnknownVtbl>(texture).query_interface)(
            texture,
            &IID_IDXGI_RESOURCE1,
            &mut resource,
        )
    };
    if hr < 0 || resource.is_null() {
        return Err(hresult_error("QueryInterface(IDXGIResource1)", hr));
    }
    let mut handle: Handle = NULL_HANDLE;
    let hr = unsafe {
        (vtbl::<IDXGIResource1Vtbl>(resource).create_shared_handle)(
            resource,
            std::ptr::null(),
            GENERIC_ALL,
            std::ptr::null(),
            &mut handle,
        )
    };
    unsafe { com_release(resource) };
    if hr < 0 || handle.is_null() {
        return Err(hresult_error("IDXGIResource1::CreateSharedHandle", hr));
    }
    Ok(handle)
}

/// mpv's GL loader for the hidden context, and this module's own.
/// Windows needs both halves of the well-known two-step: extensions
/// (everything framebuffer- and interop-related) resolve only through
/// `wglGetProcAddress` against the current context, while the GL 1.1
/// core that opengl32.dll exports resolves only through
/// `GetProcAddress` — and some drivers answer 1/2/3/-1 rather than NULL
/// from `wglGetProcAddress` for the latter, so those sentinels are
/// filtered too. A loader doing only one half silently fails mpv on half
/// its symbol requests (the Windows twin of the README's libepoxy note).
pub(crate) fn gl_proc_address(name: &str) -> *mut c_void {
    let Ok(cname) = CString::new(name) else {
        return std::ptr::null_mut();
    };
    let ptr = unsafe { wglGetProcAddress(cname.as_ptr()) };
    if !matches!(ptr as isize, -1..=3) {
        return ptr;
    }
    static OPENGL32: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    // opengl32 is linked into the process, so it is already loaded.
    let module =
        *OPENGL32.get_or_init(|| unsafe { GetModuleHandleA(c"opengl32.dll".as_ptr()) as usize });
    if module == 0 {
        return std::ptr::null_mut();
    }
    unsafe { GetProcAddress(module as Hmodule, cname.as_ptr()) }
}
