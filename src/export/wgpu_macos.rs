//! Zero-copy wgpu import for [`ExportedFrame`] (`wgpu` feature, macOS):
//! IOSurface → `MTLTexture` on the consumer's device → `wgpu::Texture`
//! through wgpu-hal — no pixel copies anywhere.
//!
//! The lifetime story is the point of this module: the pool buffer moves
//! into wgpu-hal's texture drop callback, and wgpu releases a hal
//! texture only once the GPU has finished every submission using it. So
//! the pool cannot recycle the IOSurface under in-flight GPU work — the
//! manual "hold the frame until the completion handler" obligation that
//! [`ExportedFrame::io_surface`] documents disappears entirely on this
//! path.
//!
//! Everything here is wgpu-major-locked churn (the README's roadmap
//! reason for a separate non-default feature): `as_hal`'s guard shape,
//! `texture_from_raw`'s parameter list, and the objc2 generation
//! wgpu-hal's Metal backend speaks all move with wgpu majors.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use crate::error::{Error, Result};

use super::ExportedFrame;

impl ExportedFrame {
    /// Import this frame into `device` as a zero-copy [`wgpu::Texture`]
    /// (`Bgra8Unorm`, `TEXTURE_BINDING | COPY_SRC`), consuming the
    /// frame: the underlying IOSurface aliases the texture's memory, and
    /// the frame's pool buffer is returned only when wgpu releases the
    /// texture — which wgpu does after the GPU has finished all work
    /// using it. Unlike the raw [`io_surface`](Self::io_surface) path
    /// there is no reuse hazard to manage: drop the texture whenever
    /// you're done encoding with it.
    ///
    /// Import a fresh texture per acquired frame (it's a cheap wrap, not
    /// a copy). Errors with [`Error::WgpuImport`] when `device` isn't on
    /// the Metal backend or the driver refuses the IOSurface wrap; the
    /// frame is consumed either way (its buffer returns to the pool on
    /// the error path).
    pub fn into_wgpu_texture(mut self, device: &wgpu::Device) -> Result<wgpu::Texture> {
        let (width, height) = (self.width(), self.height());
        // Clone the Retained MTLDevice out and release the as_hal guard
        // immediately — holding it across other wgpu calls on the same
        // device is asking for a lock-order surprise.
        let raw_device = {
            // SAFETY: read-only access to the hal device; the guard is
            // dropped before any other wgpu call.
            let hal_device = unsafe { device.as_hal::<wgpu_hal::api::Metal>() };
            let Some(hal_device) = hal_device else {
                return Err(Error::WgpuImport(
                    "wgpu device is not on the Metal backend".into(),
                ));
            };
            hal_device.raw_device().clone()
        };

        // SAFETY: descriptor construction with valid dimensions.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        // IOSurface-backed textures: Shared on unified-memory GPUs
        // (Apple silicon), Managed on discrete ones — Shared textures
        // don't exist on Intel-era macOS GPUs.
        descriptor.setStorageMode(if raw_device.hasUnifiedMemory() {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        // SAFETY: `io_surface()` is a live retained IOSurfaceRef for as
        // long as `self` is (`self.buffer` is still in place here), and
        // the created texture retains the surface it wraps.
        let raw_texture = unsafe {
            let surface = &*self.io_surface().cast::<IOSurfaceRef>();
            raw_device.newTextureWithDescriptor_iosurface_plane(&descriptor, surface, 0)
        };
        drop::<Retained<ProtocolObject<dyn MTLDevice>>>(raw_device);
        let Some(raw_texture) = raw_texture else {
            return Err(Error::WgpuImport(format!(
                "newTextureWithDescriptor:iosurface:plane: failed for {width}x{height}"
            )));
        };

        // Move the pool buffer into hal's drop callback: it fires when
        // wgpu is done with the texture, GPU work included — exactly the
        // moment the pool may recycle the surface. Our own Drop sees
        // `None` and stands down.
        let buffer = self
            .buffer
            .take()
            .expect("buffer present until drop/presented");
        let shared = Arc::clone(&self.shared);
        let drop_callback: wgpu_hal::DropCallback = Box::new(move || {
            super::return_buffer(&shared, buffer);
        });

        // SAFETY: the texture is valid, matches the descriptor below
        // (2D, one mip, one layer, Bgra8Unorm), and is initialized —
        // mpv rendered into it before publish.
        let hal_texture = unsafe {
            wgpu_hal::metal::Device::texture_from_raw(
                raw_texture,
                wgpu::TextureFormat::Bgra8Unorm,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu_hal::CopyExtent {
                    width,
                    height,
                    depth: 1,
                },
                Some(drop_callback),
            )
        };
        let desc = wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        // SAFETY: hal texture and descriptor agree, and the content is
        // fully initialized (RESOURCE = shader-readable).
        Ok(unsafe {
            device.create_texture_from_hal::<wgpu_hal::api::Metal>(
                hal_texture,
                &desc,
                wgpu::TextureUses::RESOURCE,
            )
        })
    }
}
