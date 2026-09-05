//! wgpu import tests: exported frames wrapped as `wgpu::Texture`s on a
//! real Metal device, verified by reading the texture back through wgpu
//! itself. Skips like the other suites when mpv, ffmpeg, a GL context,
//! or a wgpu adapter is unavailable.
#![cfg(all(feature = "wgpu", target_os = "macos"))]

mod common;

use common::engine_with_frame;

fn wgpu_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = match pollster::block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(e) => {
            eprintln!("skipping: no wgpu Metal adapter: {e}");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&Default::default())) {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("skipping: wgpu device unavailable: {e}");
            None
        }
    }
}

/// Read a 64x64 Bgra8 texture back through wgpu (64 * 4 = 256 bytes per
/// row — exactly wgpu's row alignment, no padding to strip).
fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u8> {
    let size = 64 * 64 * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    drop(buffer);
    data
}

#[test]
fn imported_texture_matches_frame_pixels() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    let Some((device, queue)) = wgpu_device() else {
        return;
    };
    let expected = frame.copy_pixels();
    let texture = frame.into_wgpu_texture(&device).expect("wgpu import");
    assert_eq!(texture.format(), wgpu::TextureFormat::Bgra8Unorm);
    assert_eq!((texture.width(), texture.height()), (64, 64));

    let via_wgpu = read_back(&device, &queue, &texture);
    assert_eq!(via_wgpu.len(), expected.len());
    // Identical bytes: zero-copy means the texture *is* the IOSurface.
    assert_eq!(via_wgpu, expected, "wgpu readback differs from IOSurface");
    // And the content sanity from the clip (BGRA, red top / black
    // bottom), so a doubly-wrong path can't pass by agreeing with
    // itself.
    let top = (8 * 64 + 32) * 4;
    assert!(
        via_wgpu[top + 2] > 150 && via_wgpu[top] < 100,
        "top pixel should be red in BGRA"
    );

    // Dropping the texture must release the pool buffer (the hal drop
    // callback) without deadlock, and detach must still tear down clean.
    drop(texture);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    engine.detach_render();
}

#[test]
fn texture_outlives_frame_and_detach() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    let Some((device, queue)) = wgpu_device() else {
        return;
    };
    let expected = frame.copy_pixels();
    let texture = frame.into_wgpu_texture(&device).expect("wgpu import");
    // The frame is consumed and the engine detached — the texture (and
    // the IOSurface it retains) must remain fully readable: the drop
    // callback owns the buffer now, not the pool or the render thread.
    engine.detach_render();
    drop(engine);
    let via_wgpu = read_back(&device, &queue, &texture);
    assert_eq!(via_wgpu, expected);
}
