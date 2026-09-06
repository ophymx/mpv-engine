//! Zero-copy wgpu import for [`ExportedFrame`] (`wgpu` feature,
//! Windows): shared D3D11 texture → `ID3D12Resource` on the consumer's
//! device → `wgpu::Texture` through wgpu-hal — no pixel copies
//! anywhere, and no hand-rolled D3D12 in this crate beyond the one
//! `OpenSharedHandle` call.
//!
//! The lifetime story follows the Linux path, not the Metal one.
//! wgpu-hal's `dx12::Device::texture_from_raw` owns the resource it is
//! handed and offers no drop-callback seam, while the opened
//! `ID3D12Resource` holds its own reference on the shared memory — so
//! instead of returning the pool buffer when wgpu finishes, the import
//! *retires* it ([`super::retire_buffer`]): the buffer leaves the pool
//! for good, mpv never renders into that memory again (killing the
//! aliasing hazard the same way the Metal drop callback does), the
//! render thread allocates a replacement, and wgpu frees the resource on
//! its own completion schedule. The consumer-facing contract is
//! identical on all three platforms: drop the texture whenever you're
//! done encoding.
//!
//! No fence machinery: the backend's publish barrier is a `glFinish`
//! plus the interop unlock (see the platform module), so a frame's
//! pixels are complete and the texture is back in D3D's hands before the
//! shell can ever see the frame. The imported resource is declared to
//! wgpu in `COMMON` — the state a cross-API shared resource is always in
//! at a submission boundary — so wgpu's first barrier transitions out of
//! it correctly rather than from a state D3D12 never saw.
//!
//! Everything here is wgpu-major-locked churn (the README's roadmap
//! reason for a separate non-default feature): `as_hal`'s guard shape,
//! `texture_from_raw`'s parameter list, and the `windows` generation
//! wgpu-hal's DX12 backend speaks all move with wgpu majors.

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::error::{Error, Result};

use super::ExportedFrame;

impl ExportedFrame {
    /// Import this frame into `device` as a zero-copy [`wgpu::Texture`]
    /// (`Bgra8Unorm`, `TEXTURE_BINDING | COPY_SRC`), consuming the
    /// frame: the texture's memory *is* the frame's shared D3D11
    /// texture, opened on the consumer's D3D12 device, and wgpu releases
    /// it only after the GPU has finished all work using the texture.
    /// Unlike the raw [`shared_handle`](Self::shared_handle) path there
    /// is no reuse hazard to manage: drop the texture whenever you're
    /// done encoding with it.
    ///
    /// Import a fresh texture per acquired frame (it's an open, not a
    /// copy). `device` must be on the DX12 backend and on the same
    /// adapter the engine's hidden GL context landed on — a shared
    /// handle does not cross GPUs, so a mismatch surfaces here as an
    /// [`Error::WgpuImport`], not as corruption. The same error covers a
    /// non-DX12 device and a driver refusing the open; the frame is
    /// consumed either way (its buffer returns to the pool on the error
    /// path).
    pub fn into_wgpu_texture(mut self, device: &wgpu::Device) -> Result<wgpu::Texture> {
        let (width, height) = (self.width(), self.height());
        let handle = HANDLE(self.shared_handle());
        let resource = {
            // The guard is read-only access to the hal device and is
            // dropped before `create_texture_from_hal` (a wgpu-core call
            // on the same device) below.
            //
            // SAFETY: read-only access to the hal device.
            let hal_device = unsafe { device.as_hal::<wgpu_hal::api::Dx12>() };
            let Some(hal_device) = hal_device else {
                return Err(Error::WgpuImport(
                    "wgpu device is not on the DX12 backend".into(),
                ));
            };
            let mut resource: Option<ID3D12Resource> = None;
            // SAFETY: `handle` is a live shared NT handle to a D3D11
            // texture (`self.buffer` is still in place here), and
            // `OpenSharedHandle` duplicates it internally rather than
            // taking ownership.
            unsafe {
                hal_device
                    .raw_device()
                    .OpenSharedHandle(handle, &mut resource)
            }
            .map_err(|e| {
                Error::WgpuImport(format!("ID3D12Device::OpenSharedHandle failed: {e}"))
            })?;
            resource
                .ok_or_else(|| Error::WgpuImport("OpenSharedHandle returned no resource".into()))?
        };

        // SAFETY: the resource is valid, matches the descriptor below
        // (2D, one mip, one sample, Bgra8Unorm), and is initialized —
        // mpv rendered into it and the publish barrier finished before
        // the frame was published.
        let hal_texture = unsafe {
            wgpu_hal::dx12::Device::texture_from_raw(
                resource,
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureDimension::D2,
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                1,
                1,
            )
        };

        // The open succeeded: wgpu owns a D3D12 resource holding its own
        // reference on the shared memory, so the pool buffer is retired —
        // mpv must never render into that memory again. Our own Drop
        // sees `None` and stands down.
        let buffer = self
            .buffer
            .take()
            .expect("buffer present until drop/presented");
        super::retire_buffer(&self.shared, buffer);

        let desc = super::exported_texture_desc(width, height);
        // SAFETY: hal texture and descriptor agree, and the content is
        // fully initialized. The empty initial state is deliberate: it
        // maps to `D3D12_RESOURCE_STATE_COMMON`, which is the state a
        // shared resource is in at every submission boundary — declaring
        // `RESOURCE` here would make wgpu's first barrier claim a
        // before-state D3D12 never put the resource in.
        Ok(unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Dx12>(
                hal_texture,
                &desc,
                wgpu::TextureUses::empty(),
            )
        })
    }
}
