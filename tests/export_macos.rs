//! Exported-frame backend tests against real libmpv and a real (hidden)
//! CGL context — macOS + `export` feature only. Like the headless suite,
//! missing tooling degrades to a skip: no ffmpeg, no libmpv, or no
//! WindowServer/GPU access (SSH, bare CI) all print and return instead
//! of failing.
#![cfg(all(feature = "export", target_os = "macos"))]

use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mpv_engine::{Engine, ExportOptions, ExportedFrame};

/// A render-API (`vo=libmpv`) engine, or None when mpv is unavailable.
/// Audio nulled for the same reason as in the headless suite: the AO
/// probe can hard-abort a box with no sound server.
fn video_engine() -> Option<Engine> {
    match Engine::video().property("ao", "null").build() {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping: mpv engine unavailable: {e}");
            None
        }
    }
}

/// 1 second of 64x64 video: top half red, bottom half black (rawvideo in
/// NUT — no encoder needed). One clip pins the whole seam: pixels
/// arriving at all, BGRA byte order, and vertical orientation (row 0
/// must be the top). None when ffmpeg is absent.
fn generate_red_over_black(target: &Path) -> Option<()> {
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=64x32:rate=10",
            "-vf",
            "pad=64:64:0:0:black",
            "-t",
            "1",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(target)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => Some(()),
        _ => {
            eprintln!("skipping: ffmpeg unavailable or failed");
            None
        }
    }
}

/// Engine with the exported backend attached, the test clip loaded, and
/// the first published frame *with video content* acquired — early
/// publishes can be the pre-load clear render, so all-black frames are
/// released back and waited past. None on any skip condition. The
/// receiver carries the backend's frame-published signals.
fn engine_with_frame() -> Option<(Engine, ExportedFrame, mpsc::Receiver<()>)> {
    let engine = video_engine()?;
    let dir = tempfile::tempdir().expect("tempdir");
    let clip = dir.path().join("red_over_black.nut");
    generate_red_over_black(&clip)?;
    let (tx, rx) = mpsc::channel();
    if let Err(e) = engine.attach_exported_render(ExportOptions::new(64, 64), move || {
        let _ = tx.send(());
    }) {
        eprintln!("skipping: exported render unavailable: {e}");
        return None;
    }
    engine
        .load_when_ready(clip.to_str().expect("utf-8 temp path"))
        .expect("load");
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        assert!(
            Instant::now() < deadline,
            "no exported frame published within 15s"
        );
        // Signals can race the acquire (newest-wins pool), so treat them
        // as hints and poll the slot.
        let _ = rx.recv_timeout(Duration::from_millis(200));
        if let Some(frame) = engine.acquire_frame().expect("acquire") {
            // Alpha is opaque even on the pre-load clear render — only
            // the color channels distinguish video content.
            let has_content = frame
                .copy_pixels()
                .chunks_exact(4)
                .any(|px| px[0] | px[1] | px[2] != 0);
            if has_content {
                return Some((engine, frame, rx));
            }
        }
    }
}

#[test]
fn exported_frames_carry_correctly_oriented_bgra_pixels() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    assert_eq!((frame.width(), frame.height()), (64, 64));
    assert!(!frame.io_surface().is_null());

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
    // Detach (joins the render thread, destroys the GL context) with the
    // frame still in hand: the IOSurface must stay alive and readable,
    // and nothing may deadlock or crash — including the frame's return
    // to the torn-down pool afterwards, and dropping the engine last.
    engine.detach_render();
    let px = frame.copy_pixels();
    assert_eq!(px.len(), 64 * 64 * 4);
    drop(engine);
    drop(frame);
}
