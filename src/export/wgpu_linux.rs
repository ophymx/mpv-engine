//! Zero-copy wgpu import for [`ExportedFrame`] (`wgpu` feature, Linux):
//! DMA-BUF → `VkImage` on the consumer's device → `wgpu::Texture`,
//! through wgpu-hal's own dmabuf import (`texture_from_dmabuf_fd`,
//! DRM-modifier tiling with our linear layout) — no pixel copies
//! anywhere, and no hand-rolled ash in this crate.
//!
//! The lifetime story deliberately differs from the Metal path.
//! wgpu-hal's dmabuf import owns the imported image and memory outright
//! (there is no drop-callback seam on that path), and the imported
//! memory holds its own kernel reference on the dmabuf — so instead of
//! returning the pool buffer when wgpu finishes, the import *retires*
//! it ([`super::retire_buffer`]): the buffer leaves the pool for good,
//! mpv never renders into that memory again (killing the aliasing
//! hazard the same way the Metal drop callback does), the render thread
//! allocates a replacement, and wgpu frees image + memory on its own
//! completion schedule. The consumer-facing contract is identical on
//! both platforms: drop the texture whenever you're done encoding.
//!
//! No semaphore machinery: the backend's publish barrier is a
//! `glFinish` (see the platform module), so a frame's pixels are
//! complete before the shell can ever see the frame. The roadmap's
//! sync-fd tier is what would put `add_wait_semaphore` back on the
//! table here.

use crate::error::{Error, Result};

use super::{ExportedFrame, platform};

impl ExportedFrame {
    /// Import this frame into `device` as a zero-copy [`wgpu::Texture`]
    /// (`Bgra8Unorm`, `TEXTURE_BINDING | COPY_SRC`), consuming the
    /// frame: the texture's memory *is* the frame's DMA-BUF, imported
    /// into Vulkan, and wgpu releases it only after the GPU has
    /// finished all work using the texture. Unlike the raw
    /// [`dma_buf_fd`](Self::dma_buf_fd) path there is no reuse hazard
    /// to manage: drop the texture whenever you're done encoding with
    /// it.
    ///
    /// Import a fresh texture per acquired frame (it's an import, not a
    /// copy). `device` must be on the Vulkan backend and created with
    /// [`wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF`] in its
    /// `required_features` (wgpu enables the underlying
    /// `VK_EXT_external_memory_dma_buf` / `VK_EXT_image_drm_format_modifier`
    /// extensions whenever the driver offers them — the feature flag is
    /// how the consumer opts in). Errors with [`Error::WgpuImport`]
    /// otherwise, or when the driver refuses the import; the frame is
    /// consumed either way (its buffer returns to the pool on the error
    /// path).
    ///
    /// On an `XR24` frame (see [`fourcc`](Self::fourcc)) the texture is
    /// still `Bgra8Unorm`; treat its alpha channel as undefined.
    pub fn into_wgpu_texture(mut self, device: &wgpu::Device) -> Result<wgpu::Texture> {
        let (width, height) = (self.width(), self.height());
        if !device
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        {
            return Err(Error::WgpuImport(
                "device lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF — request it in required_features \
                 (needs driver support for VK_EXT_external_memory_dma_buf and \
                 VK_EXT_image_drm_format_modifier)"
                    .into(),
            ));
        }
        let fourcc = self.fourcc();
        if fourcc != platform::FOURCC_ARGB8888 && fourcc != platform::FOURCC_XRGB8888 {
            return Err(Error::WgpuImport(format!(
                "unsupported DRM fourcc {fourcc:#010x} (expected AR24/XR24)"
            )));
        }
        // Vulkan takes ownership of the fd it imports, so hand it a
        // duplicate; the frame's own fd stays with the pool buffer.
        let fd = self
            .dma_buf_fd()
            .try_clone_to_owned()
            .map_err(|e| Error::WgpuImport(format!("duplicating the dmabuf fd: {e}")))?;
        let stride = u64::from(self.stride());

        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("mpv-exported-frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let hal_texture = {
            // The guard is read-only access to the hal device and is
            // dropped before `create_texture_from_hal` (a wgpu-core call
            // on the same device) below.
            let hal_device = unsafe { device.as_hal::<wgpu_hal::api::Vulkan>() };
            let Some(hal_device) = hal_device else {
                return Err(Error::WgpuImport(
                    "wgpu device is not on the Vulkan backend".into(),
                ));
            };
            // SAFETY: `fd` is a valid dmabuf matching `hal_desc`, with
            // the linear modifier, `stride` and offset 0 straight from
            // the buffer's GBM allocation. Ownership of `fd` transfers
            // (it is our duplicate).
            unsafe {
                hal_device.texture_from_dmabuf_fd(
                    fd,
                    &hal_desc,
                    platform::MODIFIER_LINEAR,
                    stride,
                    0,
                )
            }
            .map_err(|e| Error::WgpuImport(format!("dmabuf import failed: {e}")))?
        };

        // The import succeeded: wgpu owns the image and the imported
        // memory (its own kernel reference on the dmabuf), so the pool
        // buffer is retired — mpv must never render into that memory
        // again. Our own Drop sees `None` and stands down.
        let buffer = self
            .buffer
            .take()
            .expect("buffer present until drop/presented");
        super::retire_buffer(&self.shared, buffer);

        let desc = super::exported_texture_desc(width, height);
        // SAFETY: hal texture and descriptor agree, and the content is
        // fully initialized (RESOURCE = shader-readable) — mpv rendered
        // and glFinish'd before publish. (The image's *layout* was never
        // touched by Vulkan; for a linear DRM-modifier image the
        // declared-vs-actual mismatch on the first transition is
        // content-preserving, which the byte-for-byte import test pins.)
        Ok(unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(
                hal_texture,
                &desc,
                wgpu::TextureUses::RESOURCE,
            )
        })
    }
}
