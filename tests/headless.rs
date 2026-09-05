//! Engine tests against real libmpv, headless (`vo=null`) — no display,
//! no toolkit. Media inputs are generated with ffmpeg; tests skip
//! gracefully when it (or an mpv build) is unavailable, so a bare CI box
//! degrades to a no-op instead of a failure.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use mpv_engine::{EndReason, Engine, PlaybackEvent, PropertyFormat, PropertyValue};

fn headless_engine() -> Option<Engine> {
    match Engine::headless().build() {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping: mpv engine unavailable: {e}");
            None
        }
    }
}

/// 1 second of silence in an ogg container, or None when ffmpeg is absent.
fn generate_audio(target: &Path) -> Option<()> {
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            "1",
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

/// 1 second of 64x64 test video (rawvideo in NUT — no encoder needed),
/// or None when ffmpeg is absent.
fn generate_video(target: &Path) -> Option<()> {
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10",
            "-t",
            "1",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
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

fn pump_until(engine: &Engine, mut stop: impl FnMut(&PlaybackEvent) -> bool) -> bool {
    for _ in 0..100 {
        for ev in engine.pump_events() {
            if stop(&ev) {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// The spaced-path regression that motivated the array-args command layer:
/// a file whose name carries spaces, quotes, and parens must reach
/// `Loaded`. (Under a string-joined command layer this failed with
/// `MPV_ERROR_INVALID_PARAMETER`.)
#[test]
fn loadfile_handles_awkward_filenames() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("Riley Cyriis_test \"q\" (copy).ogg");
    if generate_audio(&source).is_none() {
        return;
    }
    let Some(engine) = headless_engine() else {
        return;
    };
    engine.load(source.to_str().unwrap()).unwrap();

    let loaded = pump_until(&engine, |ev| match ev {
        PlaybackEvent::Loaded => true,
        PlaybackEvent::Failed { message, .. } => {
            panic!("loadfile failed on awkward path: {message}")
        }
        _ => false,
    });
    assert!(loaded, "awkward-path file never reached Loaded");
}

/// An empty/invalid file must surface a typed `Failed` event rather than
/// being swallowed. A 0-byte file is rejected at demux, before any
/// audio-output init, so this doesn't need a sound device either.
#[test]
fn empty_file_surfaces_failed_event() {
    let Some(engine) = headless_engine() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("empty.mkv");
    std::fs::File::create(&bad).unwrap();
    engine.load(bad.to_str().unwrap()).unwrap();

    let failed = pump_until(&engine, |ev| match ev {
        PlaybackEvent::Failed { code, .. } => {
            // mpv error codes are negative by contract — integrators rely
            // on the raw code to map to their own error copy.
            assert!(*code < 0, "Failed event must carry a negative mpv code");
            true
        }
        _ => false,
    });
    assert!(failed, "empty file must produce a Failed playback event");
}

/// `load_paused` really holds playback: after `Loaded`, the engine reports
/// paused and position does not advance until unpaused.
#[test]
fn load_paused_holds_until_unpaused() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("tone.ogg");
    if generate_audio(&source).is_none() {
        return;
    }
    let Some(engine) = headless_engine() else {
        return;
    };
    engine.load_paused(source.to_str().unwrap()).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Loaded
    )));
    assert!(engine.is_paused(), "engine must come up paused");

    // Even while paused, time-pos appears a beat after `Loaded` and can
    // shift once more as the initial playback restart settles. Wait for
    // two consecutive equal samples 200ms apart — that pair *is* the
    // hold check: a genuinely advancing position never stabilizes.
    let mut held = None;
    let settled = (0..25).any(|_| {
        let a = engine.position();
        std::thread::sleep(Duration::from_millis(200));
        if a.is_some() && a == engine.position() {
            held = a;
            true
        } else {
            false
        }
    });
    assert!(settled, "position must settle and hold while paused");

    engine.set_paused(false).unwrap();
    assert!(!engine.is_paused());
    let advanced = (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(50));
        engine.position() > held
    });
    assert!(advanced, "position must advance after unpausing");
}

/// `stop()` surfaces as `Ended { reason: Stop }` — the distinction
/// playlist logic needs (user stop must not auto-advance, EOF should).
#[test]
fn stop_reports_stop_reason() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("tone.ogg");
    if generate_audio(&source).is_none() {
        return;
    }
    let Some(engine) = headless_engine() else {
        return;
    };
    engine.load_paused(source.to_str().unwrap()).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Loaded
    )));

    engine.stop().unwrap();
    let ended = pump_until(&engine, |ev| match ev {
        PlaybackEvent::Ended { reason } => {
            assert_eq!(*reason, EndReason::Stop, "stop must not look like EOF");
            true
        }
        _ => false,
    });
    assert!(ended, "stop must produce an Ended event");
}

/// Observation delivers the current value immediately, then pushes
/// changes: no polling needed for UI state like volume sliders.
#[test]
fn observe_pushes_initial_value_and_changes() {
    let Some(engine) = headless_engine() else {
        return;
    };
    let observe_id = engine.observe("volume", PropertyFormat::Double).unwrap();

    // mpv sends the current value right after observe registration, and
    // the event's id must round-trip so multiple observations of one
    // property stay distinguishable.
    let initial = pump_until(&engine, |ev| {
        matches!(
            ev,
            PlaybackEvent::PropertyChanged { id, name, .. }
                if name == "volume" && *id == observe_id
        )
    });
    assert!(initial, "observe must push the initial value with its id");

    engine.set_volume(55.0).unwrap();
    let changed = pump_until(&engine, |ev| {
        matches!(
            ev,
            PlaybackEvent::PropertyChanged { name, value: PropertyValue::Double(v), .. }
                if name == "volume" && *v == 55.0
        )
    });
    assert!(changed, "volume change must arrive as PropertyChanged");
}

/// The software render path end-to-end, no GL and no display: attach,
/// load real video, wait for a frame signal, render, and pin the
/// opaque-alpha guarantee plus the backend-mismatch error.
#[test]
fn sw_render_produces_opaque_rgba_frames() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let engine = match Engine::video().build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping: mpv engine unavailable: {e}");
            return;
        }
    };

    // With no backend attached, update processing is a `false` no-op.
    assert!(!engine.render_update());

    let frame_ready = Arc::new(AtomicBool::new(false));
    let flag = frame_ready.clone();
    engine
        .attach_sw_render(move || flag.store(true, Ordering::SeqCst))
        .unwrap();
    // The registration itself signals once synchronously; clear that so
    // the flag below means "a real frame wants drawing".
    frame_ready.store(false, Ordering::SeqCst);

    engine.load(source.to_str().unwrap()).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Loaded
    )));
    let ready = (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(50));
        frame_ready.load(Ordering::SeqCst)
    });
    assert!(ready, "update callback must signal a frame");

    // The signaled frame must also be visible through the pull side of
    // the seam: `render_update` reports a frame wants drawing.
    let update_frame = (0..100).any(|_| {
        if engine.render_update() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
        false
    });
    assert!(update_frame, "render_update must report the pending frame");

    let mut buf = Vec::new();
    engine.render_sw(64, 64, &mut buf).unwrap();
    assert_eq!(buf.len(), 64 * 64 * 4);
    assert!(
        buf.chunks_exact(4).all(|px| px[3] == 0xFF),
        "every pixel's alpha must be forced opaque"
    );
    assert!(
        buf.chunks_exact(4)
            .any(|px| px[..3].iter().any(|&b| b != 0)),
        "a rendered test-pattern frame must contain non-black pixels"
    );

    // A GL draw against the software backend is a wiring bug and must
    // surface as an error, not a silent no-op.
    assert!(engine.render_gl(0, 64, 64, false).is_err());

    engine.detach_render();
    assert!(!engine.has_render());
}

/// The wakeup callback pushes a signal when events queue — no polling
/// timer needed to learn about `Loaded`.
#[test]
fn wakeup_callback_fires_on_events() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("tone.ogg");
    if generate_audio(&source).is_none() {
        return;
    }
    let Some(engine) = headless_engine() else {
        return;
    };

    let woke = Arc::new(AtomicBool::new(false));
    let flag = woke.clone();
    engine.set_wakeup_callback(move || flag.store(true, Ordering::SeqCst));
    // Registration fires the callback once synchronously (and mpv may
    // wake spuriously); clear so the flag below means "events queued
    // after load", not the registration echo.
    woke.store(false, Ordering::SeqCst);

    engine.load(source.to_str().unwrap()).unwrap();
    let signaled = (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(50));
        woke.load(Ordering::SeqCst)
    });
    assert!(signaled, "wakeup must fire when events queue");
    assert!(
        pump_until(&engine, |ev| matches!(ev, PlaybackEvent::Loaded)),
        "the signaled events must include Loaded"
    );
}

/// `GlRenderOptions` must be constructible from outside the crate: it is
/// `#[non_exhaustive]`, which forbids struct expressions here (E0639 —
/// functional record update gets no exemption), so the chainable setters
/// are the consumer path and this file being an external crate makes the
/// compile itself the probe. Pure construction — no mpv needed.
#[test]
fn gl_render_options_build_externally() {
    let opts = mpv_engine::GlRenderOptions::default()
        .block_for_target_time(false)
        .advanced_control(true);
    assert!(!opts.block_for_target_time);
    assert!(opts.advanced_control);
}

/// Engines drop cleanly without a render context ever attached (the
/// explicit Drop-order path with an empty render slot).
#[test]
fn engine_drops_cleanly_without_render() {
    let Some(engine) = headless_engine() else {
        return;
    };
    assert!(!engine.has_render());
    drop(engine);
}

/// A `quit` through the command escape hatch must surface as a
/// `Shutdown` event — the shell's only direct signal that the core is
/// gone (an `Ended { reason: Quit }` only accompanies it when a file was
/// playing).
#[test]
fn quit_surfaces_shutdown_event() {
    let Some(engine) = headless_engine() else {
        return;
    };
    engine.command("quit", &[]).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Shutdown
    )));
}
