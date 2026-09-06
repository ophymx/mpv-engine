//! Zero-copy wgpu import for [`ExportedFrame`] (`wgpu` feature, Linux):
//! DMA-BUF → `VkImage` on the consumer's device → `wgpu::Texture` — no
//! pixel copies anywhere.
//!
//! **POC (github.com/ophymx/mpv-engine/issues/4):** this hand-rolls the
//! dmabuf → `VkImage` import with `ash` and wraps it via wgpu-hal's
//! *public* `texture_from_raw`, rather than the convenience
//! `texture_from_dmabuf_fd`. The reason is the lifetime story: the
//! convenience path hardcodes `drop_callback: None`, taking ownership of
//! the image with no completion hook — which forced Linux to *retire* the
//! pool buffer on every imported frame and reallocate a fresh GBM bo +
//! EGLImage (~33 MB at 4K, per displayed frame). `texture_from_raw`
//! accepts a `DropCallback`; wgpu fires it only once every submission
//! using the texture has retired — exactly when the pool may recycle — so
//! the buffer is **returned** and reused, the same as the Metal path and
//! the Windows `ReleaseKeeper` path. `TextureMemory::External` keeps the
//! image and imported memory ours to destroy in the callback.
//!
//! The consumer-facing contract is unchanged and identical across
//! platforms: drop the texture whenever you're done encoding with it.
//!
//! No semaphore machinery: the backend's publish barrier is a `glFinish`
//! (see the platform module), so a frame's pixels are complete before the
//! shell can ever see the frame. The roadmap's sync-fd tier is what would
//! put `add_wait_semaphore` back on the table here.

use std::os::fd::IntoRawFd;
use std::sync::Arc;

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
        // POC (github.com/ophymx/mpv-engine/issues/4): wgpu-hal's
        // `texture_from_dmabuf_fd` hardcodes `drop_callback: None`, so it
        // owns the imported image with no completion hook — which forced the
        // retire-and-reallocate model on Linux (a fresh GBM bo + EGLImage per
        // displayed frame). Build the same VkImage + imported memory directly
        // and wrap it with the *public* `texture_from_raw`, passing a
        // `DropCallback`. wgpu destroys the texture — and fires the callback —
        // only once every submission using it has retired, exactly when the
        // pool may recycle, so the buffer is **returned**, not retired, the
        // same as the macOS/Metal drop-callback path.
        use ash::vk;

        let hal_texture = {
            // Read-only guard on the hal device; dropped before the
            // `create_texture_from_hal` wgpu-core call below. The cloned
            // `ash::Device` moved into the callback keeps working afterwards.
            let hal_device = unsafe { device.as_hal::<wgpu_hal::api::Vulkan>() };
            let Some(hal_device) = hal_device else {
                return Err(Error::WgpuImport(
                    "wgpu device is not on the Vulkan backend".into(),
                ));
            };
            let raw_device = hal_device.raw_device().clone();
            let phys = hal_device.raw_physical_device();
            let instance = hal_device.shared_instance().raw_instance();
            let ext_fd = ash::khr::external_memory_fd::Device::new(instance, &raw_device);

            // Create the VkImage (external memory + explicit DRM modifier),
            // mirroring wgpu-hal's own single-plane image create-info.
            let mut ext_img = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let plane = vk::SubresourceLayout {
                offset: 0,
                row_pitch: stride,
                size: 0,
                array_pitch: 0,
                depth_pitch: 0,
            };
            let mut drm = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(platform::MODIFIER_LINEAR)
                .plane_layouts(core::slice::from_ref(&plane));
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_img)
                .push_next(&mut drm);
            // SAFETY: valid create-info; `image` is destroyed on every error
            // path below and by the drop callback on success.
            let image = unsafe { raw_device.create_image(&image_info, None) }
                .map_err(|e| Error::WgpuImport(format!("vkCreateImage (dmabuf): {e}")))?;
            let reqs = unsafe { raw_device.get_image_memory_requirements(image) };

            // A successful vkAllocateMemory consumes the fd; until then it is
            // ours to close on any early return (wgpu-hal's own dance).
            let fd_raw = fd.into_raw_fd();

            let mut fd_props = vk::MemoryFdPropertiesKHR::default();
            if let Err(e) = unsafe {
                ext_fd.get_memory_fd_properties(
                    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                    fd_raw,
                    &mut fd_props,
                )
            } {
                unsafe {
                    libc::close(fd_raw);
                    raw_device.destroy_image(image, None);
                }
                return Err(Error::WgpuImport(format!(
                    "vkGetMemoryFdPropertiesKHR: {e}"
                )));
            }

            let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
            let type_bits = reqs.memory_type_bits & fd_props.memory_type_bits;
            let Some(mem_type) =
                (0..mem_props.memory_type_count).find(|i| type_bits & (1 << i) != 0)
            else {
                unsafe {
                    libc::close(fd_raw);
                    raw_device.destroy_image(image, None);
                }
                return Err(Error::WgpuImport(
                    "no Vulkan memory type accepts this dmabuf".into(),
                ));
            };

            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(fd_raw);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(mem_type)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            // SAFETY: dedicated allocation for `image`, importing `fd_raw`.
            // Vulkan owns the fd on success; on failure it does not, so close.
            let memory = match unsafe { raw_device.allocate_memory(&alloc_info, None) } {
                Ok(memory) => memory,
                Err(e) => {
                    unsafe {
                        libc::close(fd_raw);
                        raw_device.destroy_image(image, None);
                    }
                    return Err(Error::WgpuImport(format!(
                        "vkAllocateMemory (import dmabuf): {e}"
                    )));
                }
            };
            // fd is now owned by Vulkan.
            if let Err(e) = unsafe { raw_device.bind_image_memory(image, memory, 0) } {
                unsafe {
                    raw_device.free_memory(memory, None);
                    raw_device.destroy_image(image, None);
                }
                return Err(Error::WgpuImport(format!(
                    "vkBindImageMemory (dmabuf): {e}"
                )));
            }

            // Fully imported: move the pool buffer into the drop callback.
            // With `TextureMemory::External` + `Some(cb)`, wgpu-hal frees
            // neither the image nor the memory — the callback is the sole
            // owner of teardown, and it fires when wgpu is done with the
            // texture, the safe moment to recycle the buffer.
            let buffer = self
                .buffer
                .take()
                .expect("buffer present until drop/presented");
            let shared = Arc::clone(&self.shared);
            let cb_device = raw_device.clone();
            let drop_cb: wgpu_hal::DropCallback = Box::new(move || {
                // SAFETY: fires after wgpu destroyed the texture, i.e. after
                // every GPU submission using it retired; `image`/`memory` are
                // ours (External) and no longer referenced.
                unsafe {
                    cb_device.destroy_image(image, None);
                    cb_device.free_memory(memory, None);
                }
                super::return_buffer(&shared, buffer);
            });

            // SAFETY: `image` matches `hal_desc`, is backed by `memory`, and
            // both stay valid until the callback runs (External).
            unsafe {
                hal_device.texture_from_raw(
                    image,
                    &hal_desc,
                    Some(drop_cb),
                    wgpu_hal::vulkan::TextureMemory::External,
                )
            }
        };

        let desc = super::exported_texture_desc(width, height);
        // SAFETY: hal texture and descriptor agree, and the content is fully
        // initialized (RESOURCE = shader-readable) — mpv rendered and
        // glFinish'd before publish. For a linear DRM-modifier image the
        // declared-vs-actual layout mismatch on the first transition is
        // content-preserving, which the byte-for-byte import test pins.
        Ok(unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(
                hal_texture,
                &desc,
                wgpu::TextureUses::RESOURCE,
            )
        })
    }
}
