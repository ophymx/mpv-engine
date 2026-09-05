//! Shared helpers for the exported-backend test binaries (macOS +
//! `export`/`wgpu` features). Every helper degrades to a skip (`None`)
//! when tooling is missing — no ffmpeg, no libmpv, or no
//! WindowServer/GPU access.
#![allow(dead_code)]

use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mpv_engine::{Engine, ExportOptions, ExportedFrame};

/// A render-API (`vo=libmpv`) engine, or None when mpv is unavailable.
/// Audio nulled for the same reason as in the headless suite: the AO
/// probe can hard-abort a box with no sound server.
pub fn video_engine() -> Option<Engine> {
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
pub fn generate_red_over_black(target: &Path) -> Option<()> {
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
pub fn engine_with_frame() -> Option<(Engine, ExportedFrame, mpsc::Receiver<()>)> {
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
