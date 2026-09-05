//! Engine tests against real libmpv, headless (`vo=null`) — no display,
//! no toolkit. Media inputs are generated with ffmpeg; tests skip
//! gracefully when it (or an mpv build) is unavailable, so a bare CI box
//! degrades to a no-op instead of a failure.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use mpv_engine::{EndReason, Engine, PlaybackEvent, PropertyFormat, PropertyValue};

fn headless_engine() -> Option<Engine> {
    // `headless()` nulls only the video output — audio stays real because
    // audio-only playback is its point. Tests must null the audio side
    // too: on a CI box with no sound server, mpv's AO probe reaches
    // PulseAudio's client lib, which hard-aborts the whole test process
    // (`pa_mainloop_prepare(): Assertion 'm->state == STATE_PASSIVE'`).
    match Engine::headless().property("ao", "null").build() {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping: mpv engine unavailable: {e}");
            None
        }
    }
}

/// A render-API (`vo=libmpv`) engine, or None when mpv is unavailable.
/// Audio is nulled for the same reason as in [`headless_engine`]: mpv's
/// AO probe hard-aborts the process on a box with no sound server.
fn video_engine() -> Option<Engine> {
    match Engine::video().property("ao", "null").build() {
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

/// Poll `pred` at [`pump_until`]'s cadence (50ms interval, ~5s deadline)
/// until it holds — the one home for the wait/timeout policy, so a
/// flakiness fix lands everywhere at once.
fn wait_until(mut pred: impl FnMut() -> bool) -> bool {
    (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(50));
        pred()
    })
}

/// [`pump_until`]'s negative-check twin: drain events for roughly `dur`
/// at the same cadence and return everything seen — for asserting what
/// must NOT arrive inside a window.
fn pump_for(engine: &Engine, dur: Duration) -> Vec<PlaybackEvent> {
    let mut out = Vec::new();
    for _ in 0..dur.as_millis().div_ceil(50) {
        std::thread::sleep(Duration::from_millis(50));
        out.extend(engine.pump_events());
    }
    out
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
    let advanced = wait_until(|| engine.position() > held);
    assert!(advanced, "position must advance after unpausing");
}

/// On an engine whose `vo` never touches the render API, no attach is
/// coming — `load_when_ready` must degrade to a plain playing `load`
/// instead of a deferred start that would hold paused forever.
#[test]
fn load_when_ready_plays_immediately_without_render_api() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("tone.ogg");
    if generate_audio(&source).is_none() {
        return;
    }
    let Some(engine) = headless_engine() else {
        return;
    };
    engine.load_when_ready(source.to_str().unwrap()).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Loaded
    )));
    assert!(
        !engine.is_paused(),
        "no render attach is coming — playback must start immediately"
    );
}

/// The deferred-load policy end-to-end on a render-API engine: the
/// `loadfile` itself waits for the attach. Loading before the context
/// exists isn't survivable — mpv can't init the VO, drops the video
/// track, and a video-only file dies with `MPV_ERROR_NOTHING_TO_PLAY`
/// (-16) — so the pre-attach window must stay silent (no Failed), and
/// the attach call must issue the load and reach `Loaded` playing.
#[test]
fn load_when_ready_loads_on_attach() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let Some(engine) = video_engine() else {
        return;
    };
    engine.load_when_ready(source.to_str().unwrap()).unwrap();

    // The failure this API exists to prevent lands within ~100ms of an
    // eager load; 200ms of silence proves nothing was loaded eagerly.
    for ev in pump_for(&engine, Duration::from_millis(200)) {
        match ev {
            PlaybackEvent::Failed { message, .. } => {
                panic!("deferred load must not fail pre-attach: {message}")
            }
            PlaybackEvent::Loaded => panic!("load must wait for the attach"),
            _ => {}
        }
    }

    engine.attach_sw_render(|| {}).unwrap();
    let loaded = pump_until(&engine, |ev| match ev {
        PlaybackEvent::Loaded => true,
        PlaybackEvent::Failed { message, .. } => {
            panic!("deferred load failed after attach: {message}")
        }
        _ => false,
    });
    assert!(loaded, "attach must issue the deferred load");
    assert!(!engine.is_paused(), "deferred load must come up playing");
}

/// Pause intent set between `load_when_ready` and the attach carries
/// into the deferred load: the `pause` property persists across
/// `loadfile`, so the queued file comes up paused — no policy flag
/// second-guesses the user.
#[test]
fn pause_before_attach_loads_deferred_file_paused() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let Some(engine) = video_engine() else {
        return;
    };
    engine.load_when_ready(source.to_str().unwrap()).unwrap();
    engine.set_paused(true).unwrap();

    engine.attach_sw_render(|| {}).unwrap();
    assert!(pump_until(&engine, |ev| matches!(
        ev,
        PlaybackEvent::Loaded
    )));
    assert!(
        engine.is_paused(),
        "pause set before attach must hold through the deferred load"
    );
}

/// The defer-or-load decision reads the *current* `vo`, not a build-time
/// snapshot: a render-API engine switched to `vo=null` at runtime (an
/// audio-only mode) gets a plain playing load — not a source parked in
/// the queue waiting for an attach that will never come.
#[test]
fn load_when_ready_follows_runtime_vo_change() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let Some(engine) = video_engine() else {
        return;
    };
    engine.set_property("vo", "null").unwrap();
    engine.load_when_ready(source.to_str().unwrap()).unwrap();
    assert!(
        pump_until(&engine, |ev| matches!(ev, PlaybackEvent::Loaded)),
        "with vo switched off the render API, the load must not defer"
    );
    assert!(!engine.is_paused(), "playback must start immediately");
}

/// Transport commands issued through the `command` escape hatch obey the
/// same "newest call decides what plays" rule as the typed methods: a
/// `stop` between `load_when_ready` and the attach discards the queued
/// source — the attach must not resurrect it.
#[test]
fn command_stop_discards_deferred_load() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let Some(engine) = video_engine() else {
        return;
    };
    engine.load_when_ready(source.to_str().unwrap()).unwrap();
    engine.command("stop", &[]).unwrap();
    engine.attach_sw_render(|| {}).unwrap();
    // A wrongly issued load surfaces well within this window (see the
    // pre-attach check in load_when_ready_loads_on_attach).
    for ev in pump_for(&engine, Duration::from_millis(200)) {
        assert!(
            !matches!(ev, PlaybackEvent::Loaded),
            "stop before attach must discard the deferred load"
        );
    }
    assert!(engine.is_idle(), "nothing may be loaded after the stop");
}

/// An attach `Err` means "no context was attached" — nothing else. A
/// deferred source that cannot play must not fail the attach: the
/// context goes live, and the failure arrives as a `Failed` event.
#[test]
fn attach_survives_failing_deferred_load() {
    let Some(engine) = video_engine() else {
        return;
    };
    engine.load_when_ready("/nonexistent/deferred.nut").unwrap();
    engine
        .attach_sw_render(|| {})
        .expect("a failing deferred load must not fail the attach");
    assert!(engine.has_render(), "the render context must be live");
    assert!(
        pump_until(&engine, |ev| matches!(ev, PlaybackEvent::Failed { .. })),
        "the deferred load's failure must surface as a Failed event"
    );
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
    let Some(engine) = video_engine() else {
        return;
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
    let ready = wait_until(|| frame_ready.load(Ordering::SeqCst));
    assert!(ready, "update callback must signal a frame");

    // The signaled frame must also be visible through the pull side of
    // the seam: `render_update` reports a frame wants drawing.
    let update_frame = wait_until(|| engine.render_update());
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
    let signaled = wait_until(|| woke.load(Ordering::SeqCst));
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

/// `attached_render` answers "which backend?", not just "any backend?":
/// `None` before attach, the kind while attached, `None` again after
/// detach. (Only the software kind is reachable headless; the GL arm of
/// the mapping is a two-variant match pinned at the type level.)
#[test]
fn attached_render_reports_backend_kind() {
    let Some(engine) = video_engine() else {
        return;
    };
    assert_eq!(engine.attached_render(), None);

    engine.attach_sw_render(|| {}).unwrap();
    assert_eq!(
        engine.attached_render(),
        Some(mpv_engine::RenderKind::Software)
    );

    engine.detach_render();
    assert_eq!(engine.attached_render(), None);
}

/// Registering a render-update callback with no context attached is a
/// wiring bug (it could never fire) and must error loudly, not drop the
/// closure silently — before the first attach and after detach alike.
#[test]
fn render_update_callback_requires_attach() {
    let Some(engine) = video_engine() else {
        return;
    };
    assert!(matches!(
        engine.set_render_update_callback(|| {}),
        Err(mpv_engine::Error::NotAttached)
    ));

    engine.attach_sw_render(|| {}).unwrap();
    engine.set_render_update_callback(|| {}).unwrap();

    engine.detach_render();
    assert!(matches!(
        engine.set_render_update_callback(|| {}),
        Err(mpv_engine::Error::NotAttached)
    ));
}

/// `set_render_update_callback` hands mpv's update signal to the new
/// closure: the replacement is raised at registration, frame updates
/// land on it once video flows, and the attach-time closure never fires
/// again — the construct-share-register flow shells need when their
/// real callback can only capture state built after the engine.
#[test]
fn render_update_callback_replaces_attach_registration() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("test.nut");
    if generate_video(&source).is_none() {
        return;
    }
    let Some(engine) = video_engine() else {
        return;
    };

    let attach_fires = Arc::new(AtomicU32::new(0));
    let attach_counter = attach_fires.clone();
    engine
        .attach_sw_render(move || {
            attach_counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    assert!(
        attach_fires.load(Ordering::SeqCst) >= 1,
        "attach must raise its on_update synchronously"
    );

    let replacement_fires = Arc::new(AtomicU32::new(0));
    let replacement_counter = replacement_fires.clone();
    engine
        .set_render_update_callback(move || {
            replacement_counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    // The registration fire is synchronous — inside the call, like the
    // attach-time fire asserted above. A poll here would keep passing if
    // that guarantee regressed to "eventually"; a plain assert pins it.
    assert!(
        replacement_fires.load(Ordering::SeqCst) >= 1,
        "registration must raise the replacement callback synchronously"
    );
    // No async dispatch source exists before load() — nothing else can
    // move the attach counter from here on.
    let attach_count_after_swap = attach_fires.load(Ordering::SeqCst);

    engine.load(source.to_str().unwrap()).unwrap();
    let before_frames = replacement_fires.load(Ordering::SeqCst);
    let frames_signaled =
        wait_until(|| replacement_fires.load(Ordering::SeqCst) > before_frames);
    assert!(
        frames_signaled,
        "frame updates must land on the replacement callback"
    );
    assert_eq!(
        attach_fires.load(Ordering::SeqCst),
        attach_count_after_swap,
        "the replaced attach-time callback must not fire after the swap"
    );
}

/// The swap's synchronous fire runs outside every engine lock: a
/// replacement callback that queries the engine must not deadlock — the
/// regression shape for the fire-under-the-render-lock bug. (A regressed
/// engine hangs here, which is the loudest failure a deadlock can give.)
#[test]
fn render_update_callback_sync_fire_holds_no_engine_lock() {
    let Some(engine) = video_engine() else {
        return;
    };
    engine.attach_sw_render(|| {}).unwrap();
    let engine = Arc::new(engine);
    let probe = Arc::downgrade(&engine);
    let saw_context = Arc::new(AtomicBool::new(false));
    let saw = saw_context.clone();
    engine
        .set_render_update_callback(move || {
            if let Some(e) = probe.upgrade() {
                saw.store(e.has_render(), Ordering::SeqCst);
            }
        })
        .unwrap();
    assert!(
        saw_context.load(Ordering::SeqCst),
        "the sync fire must see the live context, without deadlocking"
    );
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
