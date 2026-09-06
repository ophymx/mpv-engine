//! wgpu import tests: exported frames wrapped as `wgpu::Texture`s on a
//! real device — Metal on macOS, Vulkan on Linux, DX12 on Windows —
//! verified by reading the texture back through wgpu itself. Skips like
//! the other suites when mpv, ffmpeg, a GL context, or a suitable wgpu
//! adapter is unavailable.
#![cfg(all(
    feature = "wgpu",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

mod common;

use common::engine_with_frame;
use mpv_engine::ExportedFrame;

fn wgpu_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    #[cfg(target_os = "macos")]
    {
        descriptor.backends = wgpu::Backends::METAL;
    }
    #[cfg(target_os = "linux")]
    {
        descriptor.backends = wgpu::Backends::VULKAN;
    }
    #[cfg(target_os = "windows")]
    {
        descriptor.backends = wgpu::Backends::DX12;
    }
    let instance = wgpu::Instance::new(descriptor);
    let adapter = match pollster::block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(e) => {
            eprintln!("skipping: no wgpu adapter for the native backend: {e}");
            return None;
        }
    };
    #[cfg(target_os = "linux")]
    let device_descriptor = {
        // The Linux import is behind an explicit wgpu feature opt-in.
        let needed = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
        if !adapter.features().contains(needed) {
            eprintln!("skipping: adapter lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF");
            return None;
        }
        wgpu::DeviceDescriptor {
            required_features: needed,
            ..Default::default()
        }
    };
    #[cfg(not(target_os = "linux"))]
    let device_descriptor = wgpu::DeviceDescriptor::default();
    match pollster::block_on(adapter.request_device(&device_descriptor)) {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("skipping: wgpu device unavailable: {e}");
            None
        }
    }
}

/// Whether the frame's alpha channel is storage-backed. On a Linux
/// `XR24` frame the fourth byte is undefined through the import, so the
/// byte-for-byte comparisons mask it out; everywhere else it must match
/// exactly.
fn frame_has_alpha(frame: &ExportedFrame) -> bool {
    #[cfg(target_os = "linux")]
    {
        const FOURCC_XR24: u32 = 0x3432_5258;
        frame.fourcc() != FOURCC_XR24
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = frame;
        true
    }
}

fn assert_pixels_match(via_wgpu: &[u8], expected: &[u8], has_alpha: bool, what: &str) {
    assert_eq!(via_wgpu.len(), expected.len());
    if has_alpha {
        assert_eq!(via_wgpu, expected, "{what}");
    } else {
        let mask = |px: &[u8]| {
            let mut px = px.to_vec();
            px.iter_mut().skip(3).step_by(4).for_each(|a| *a = 0);
            px
        };
        assert_eq!(mask(via_wgpu), mask(expected), "{what} (alpha masked)");
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
    let has_alpha = frame_has_alpha(&frame);
    let texture = frame.into_wgpu_texture(&device).expect("wgpu import");
    assert_eq!(texture.format(), wgpu::TextureFormat::Bgra8Unorm);
    assert_eq!((texture.width(), texture.height()), (64, 64));

    let via_wgpu = read_back(&device, &queue, &texture);
    // Identical bytes: zero-copy means the texture *is* the exported
    // buffer (IOSurface / DMA-BUF / shared D3D11 texture).
    assert_pixels_match(
        &via_wgpu,
        &expected,
        has_alpha,
        "wgpu readback differs from the exported buffer",
    );
    // And the content sanity from the clip (BGRA, red top / black
    // bottom), so a doubly-wrong path can't pass by agreeing with
    // itself.
    let top = (8 * 64 + 32) * 4;
    assert!(
        via_wgpu[top + 2] > 150 && via_wgpu[top] < 100,
        "top pixel should be red in BGRA"
    );

    // Dropping the texture must release the pool buffer (returned
    // through a hal drop callback on macOS and Linux, ReleaseKeeper on
    // Windows) without deadlock, and detach must still tear down clean.
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
    let has_alpha = frame_has_alpha(&frame);
    let texture = frame.into_wgpu_texture(&device).expect("wgpu import");
    // The frame is consumed and the engine detached — the texture (and
    // the memory it holds: retained IOSurface on macOS, imported dmabuf
    // reference on Linux, opened shared resource on Windows) must remain
    // fully readable: wgpu owns the backing now, not the pool or the
    // render thread.
    engine.detach_render();
    drop(engine);
    let via_wgpu = read_back(&device, &queue, &texture);
    assert_pixels_match(&via_wgpu, &expected, has_alpha, "readback after detach");
}

/// 2 seconds of 64x64 moving video (ffmpeg `testsrc`, rawvideo in NUT),
/// so successive frames differ. None when ffmpeg is absent.
fn generate_moving_clip(target: &std::path::Path) -> Option<()> {
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=15",
            "-t",
            "2",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(target)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(status, Ok(s) if s.success()).then_some(())
}

/// Continuously importing from a looping source must recycle pool buffers
/// *and keep them correct*. The Linux import returns each buffer to the
/// pool when wgpu is done with the texture (the same reuse macOS/Windows
/// do), so importing far more frames than the pool holds cycles every
/// buffer many times. An aliasing bug — mpv rendering into a buffer a
/// consumer still reads — would surface here two ways: a wgpu readback
/// that no longer matches the frame's own CPU copy, or content that never
/// changes (a stuck buffer). Regression guard for the reuse path
/// (github.com/ophymx/mpv-engine/issues/4).
#[test]
fn continuous_import_recycles_buffers_and_stays_correct() {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    use std::time::{Duration, Instant};

    let Some((device, queue)) = wgpu_device() else {
        return;
    };
    let Some(engine) = common::video_engine() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let clip = dir.path().join("moving.nut");
    if generate_moving_clip(&clip).is_none() {
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    if engine
        .attach_exported_render(mpv_engine::ExportOptions::new(64, 64), move || {
            let _ = tx.send(());
        })
        .is_err()
    {
        eprintln!("skipping: exported render unavailable");
        return;
    }
    engine.set_property("loop-file", "inf").expect("loop-file");
    engine
        .load_when_ready(clip.to_str().expect("utf-8 path"))
        .expect("load");

    // Far more imports than the pool size, so buffers must be recycled.
    const WANT: usize = 24;
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut imported = 0usize;
    let mut fingerprints = HashSet::new();
    while imported < WANT && Instant::now() < deadline {
        let _ = rx.recv_timeout(Duration::from_millis(100));
        let Some(frame) = engine.acquire_frame().expect("acquire") else {
            continue;
        };
        let cpu = frame.copy_pixels();
        // Skip the pre-load clear render (all black); it is not content.
        if !cpu.chunks_exact(4).any(|px| px[0] | px[1] | px[2] != 0) {
            continue;
        }
        let has_alpha = frame_has_alpha(&frame);
        let texture = frame.into_wgpu_texture(&device).expect("wgpu import");
        let via_wgpu = read_back(&device, &queue, &texture);
        assert_pixels_match(
            &via_wgpu,
            &cpu,
            has_alpha,
            "recycled-buffer readback differs from the frame's own pixels",
        );
        let mut h = std::collections::hash_map::DefaultHasher::new();
        cpu.hash(&mut h);
        fingerprints.insert(h.finish());
        drop(texture);
        // Let wgpu finish and fire the drop callback, returning the buffer
        // to the pool so the next render recycles it.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        imported += 1;
    }
    assert!(
        imported >= WANT,
        "only imported {imported}/{WANT} frames from a looping source"
    );
    assert!(
        fingerprints.len() >= 2,
        "every recycled buffer read back identical content — a stuck or aliased buffer"
    );
    engine.detach_render();
}
