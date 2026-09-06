//! Exported-frame backend tests against real libmpv and a real (hidden)
//! EGL context on a DRM render node — Linux + `export` feature only.
//! Like the headless suite, missing tooling degrades to a skip: no
//! ffmpeg, no libmpv, or no GPU/render-node access (bare CI, missing
//! render group) all print and return instead of failing.
#![cfg(all(feature = "export", target_os = "linux"))]

mod common;

use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use common::engine_with_frame;

/// DRM fourcc 'AR24' / 'XR24' — the two layouts the backend may
/// allocate (both little-endian B,G,R,A/X).
const FOURCC_AR24: u32 = 0x3432_5241;
const FOURCC_XR24: u32 = 0x3432_5258;

#[test]
fn exported_frames_carry_correctly_oriented_bgra_pixels() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    assert_eq!((frame.width(), frame.height()), (64, 64));
    assert!(matches!(frame.fourcc(), FOURCC_AR24 | FOURCC_XR24));
    assert_eq!(
        frame.modifier(),
        0,
        "backend promises DRM_FORMAT_MOD_LINEAR"
    );
    assert!(frame.stride() >= 64 * 4);
    // The dmabuf fd must be a live kernel object big enough for the
    // frame — pin it via lseek, which dmabufs support.
    let size = unsafe { libc::lseek(frame.dma_buf_fd().as_raw_fd(), 0, libc::SEEK_END) };
    assert!(
        size >= i64::from(frame.stride()) * 64,
        "dmabuf reports size {size} for stride {}",
        frame.stride()
    );

    let px = frame.copy_pixels();
    assert_eq!(px.len(), 64 * 64 * 4);
    let pixel = |x: usize, y: usize| {
        let i = (y * 64 + x) * 4;
        (px[i], px[i + 1], px[i + 2]) // BGRA layout → (B, G, R)
    };
    // Top area red, bottom black — flipped output or RGBA/BGRA confusion
    // both fail loudly here. Sampled well away from the halfway seam so
    // chroma subsampling can't blur the assertion.
    let (top_b, _top_g, top_r) = pixel(32, 8);
    assert!(
        top_r > 150 && top_b < 100,
        "top should be red in BGRA order, got B={top_b} R={top_r}"
    );
    let (bot_b, bot_g, bot_r) = pixel(32, 56);
    assert!(
        bot_b < 60 && bot_g < 60 && bot_r < 60,
        "bottom should be black, got ({bot_b},{bot_g},{bot_r})"
    );

    // `presented` consumes the frame and must not wedge anything.
    frame.presented();
    engine.detach_render();
}

#[test]
fn resize_rerenders_at_new_size_without_playback() {
    let Some((engine, first, rx)) = engine_with_frame() else {
        return;
    };
    drop(first);
    // The clip may already have ended (keep-open holds the last frame);
    // the resize must force a re-render on its own.
    engine.set_export_size(32, 32).expect("resize");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "no 32x32 frame within 10s");
        let _ = rx.recv_timeout(Duration::from_millis(200));
        match engine.acquire_frame().expect("acquire") {
            Some(frame) if (frame.width(), frame.height()) == (32, 32) => {
                assert_eq!(frame.copy_pixels().len(), 32 * 32 * 4);
                break;
            }
            _ => {}
        }
    }
    engine.detach_render();
}

#[test]
fn outstanding_frame_survives_detach() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    // Detach (joins the render thread, destroys the EGL context and GBM
    // device) with the frame still in hand: the dmabuf fd must stay
    // alive and CPU-readable (copy_pixels is pure fd/mmap — no EGL),
    // and nothing may deadlock or crash — including the frame's return
    // to the torn-down pool afterwards, and dropping the engine last.
    engine.detach_render();
    let px = frame.copy_pixels();
    assert_eq!(px.len(), 64 * 64 * 4);
    drop(engine);
    drop(frame);
}
