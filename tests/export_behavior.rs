//! Platform-independent behavior of the exported-frame backend (the
//! pool/thread logic in `src/export/mod.rs`), run against whichever
//! platform's real GL context this box has. Regression tests for review
//! findings; skips like the other export suites.
#![cfg(all(feature = "export", any(target_os = "macos", target_os = "linux")))]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use common::engine_with_frame;
use mpv_engine::{ExportOptions, ExportedFrame};

/// A continuously-updating source (a looping clip) must keep delivering
/// frames to the consumer, not just the first one. Regression for the
/// pool starving the shell: when the render thread reclaimed its own
/// just-published buffer for the next frame before the consumer's
/// acquire could win the race, the pool never grew past one live buffer
/// and every acquire after the first saw an empty slot — a black window
/// on real video. (The single-frame export suites miss this: their clip
/// ends and the render thread idles, so the last frame simply waits.)
#[test]
fn continuous_source_keeps_delivering_frames() {
    let Some(engine) = common::video_engine() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let clip = dir.path().join("loop.nut");
    // 60fps so the render thread stays saturated — a low frame rate idles
    // it between frames, and the consumer then wins the buffer race even
    // with the bug present, hiding the regression.
    if generate_fast_clip(&clip).is_none() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    if engine
        .attach_exported_render(ExportOptions::new(128, 128), move || {
            let _ = tx.send(());
        })
        .is_err()
    {
        return;
    }
    // Loop forever so updates never stop — the condition the bug needs.
    engine.set_property("loop-file", "inf").expect("loop-file");
    engine
        .load_when_ready(clip.to_str().expect("utf-8 path"))
        .expect("load");

    // Count successful acquisitions over a few seconds of playback. With
    // the pool healthy this reaches the cap almost immediately; under the
    // starvation bug it stalls at zero or one. The threshold sits far
    // below a working rate and far above the broken one, so it is not
    // timing-fragile.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut acquired = 0;
    while acquired < 10 && Instant::now() < deadline {
        let _ = rx.recv_timeout(Duration::from_millis(100));
        if engine.acquire_frame().expect("acquire").is_some() {
            acquired += 1;
        }
    }
    assert!(
        acquired >= 5,
        "continuous source starved the consumer: only {acquired} frame(s) acquired \
         (pool never grew past the published buffer?)"
    );
    engine.detach_render();
}

/// 2 seconds of 128x128 video at 60fps (rawvideo in NUT), so the export
/// render thread never idles between frames. None when ffmpeg is absent.
fn generate_fast_clip(target: &std::path::Path) -> Option<()> {
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x128:rate=60",
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

/// A forced re-render (`set_export_size`) landing while the shell holds
/// every pool buffer must not be lost: it re-arms when a frame handle
/// is released, so a paused/ended video still re-renders at the new
/// size. (Review finding: the force flag was consumed and the frame
/// silently dropped on the pool-exhausted path.)
#[test]
fn forced_render_survives_pool_exhaustion() {
    let Some((engine, first, rx)) = engine_with_frame() else {
        return;
    };
    // Pause so mpv produces no further updates of its own — any
    // re-render after this point is driven by the force alone.
    engine
        .set_property("pause", "yes")
        .expect("pause for a quiescent render thread");

    // Exhaust the default pool of 3: hold the first frame, then force a
    // render at two fresh sizes and hold each result. Wrong-size frames
    // acquired along the way (leftovers from before the pause) drop
    // straight back to the pool.
    let mut held: Vec<ExportedFrame> = vec![first];
    for size in [60u32, 56] {
        engine.set_export_size(size, size).expect("resize");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(
                Instant::now() < deadline,
                "no {size}x{size} frame while filling the pool"
            );
            let _ = rx.recv_timeout(Duration::from_millis(100));
            if let Some(frame) = engine.acquire_frame().expect("acquire") {
                if (frame.width(), frame.height()) == (size, size) {
                    held.push(frame);
                    break;
                }
            }
        }
    }
    assert_eq!(held.len(), 3, "pool should now be fully in our hands");

    // With everything held, this force has no buffer to render into and
    // the frame is dropped...
    engine
        .set_export_size(32, 32)
        .expect("resize while exhausted");
    std::thread::sleep(Duration::from_millis(300));

    // ...until releasing one frame re-arms it. Without the re-arm this
    // loop times out: the video is paused, so nothing else will ever
    // render.
    drop(held.remove(0));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(
            Instant::now() < deadline,
            "forced render was lost: no 32x32 frame after releasing a buffer"
        );
        let _ = rx.recv_timeout(Duration::from_millis(200));
        match engine.acquire_frame().expect("acquire") {
            Some(frame) if (frame.width(), frame.height()) == (32, 32) => break,
            _ => {}
        }
    }
    engine.detach_render();
}

/// Detaching from inside `on_update` — which runs on the backend's own
/// render thread — must not self-join that thread (review finding:
/// EDEADLK panic inside Drop / permanent hang). Teardown defers to just
/// after the callback returns instead.
#[test]
fn detach_from_on_update_does_not_deadlock() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    drop(frame);
    let engine = Arc::new(engine);

    // Replace the callback with one that detaches on its first fire
    // *from the render thread*. `armed` is flipped only after the
    // registration's synchronous fire (calling thread) has passed, so
    // the detach can only happen on a real publish.
    let armed = Arc::new(AtomicBool::new(false));
    let detached = Arc::new(AtomicBool::new(false));
    {
        let cb_engine = Arc::clone(&engine);
        let armed = Arc::clone(&armed);
        let detached = Arc::clone(&detached);
        engine
            .set_render_update_callback(move || {
                if armed.load(Ordering::SeqCst) && !detached.swap(true, Ordering::SeqCst) {
                    cb_engine.detach_render();
                }
            })
            .expect("replace callback");
    }
    armed.store(true, Ordering::SeqCst);

    // Force a publish; its on_update fires on the render thread and
    // detaches. A regression deadlocks or aborts right here.
    engine.set_export_size(48, 48).expect("force a publish");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !detached.load(Ordering::SeqCst) || engine.attached_render().is_some() {
        assert!(
            Instant::now() < deadline,
            "detach from on_update did not complete"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // The render thread finishes its deferred teardown on its own; give
    // it a beat, then make sure dropping the engine afterwards is clean.
    std::thread::sleep(Duration::from_millis(300));
    drop(engine);
}

/// A detach from inside `on_update` defers the render thread's teardown
/// — and parks its `JoinHandle`, so the next attach *joins* that
/// teardown instead of racing it. (Review finding: the handle was
/// discarded, leaving an immediate re-attach able to fail loudly and
/// the teardown to race process exit.) The re-attach must therefore
/// succeed first try, with no settling sleep.
#[test]
fn reattach_after_callback_detach_succeeds() {
    let Some((engine, frame, _rx)) = engine_with_frame() else {
        return;
    };
    drop(frame);
    let engine = Arc::new(engine);

    // Same arming shape as the test above: detach on the first real
    // publish, from the render thread. The lingering sleep *after* the
    // detach keeps the old thread — and the mpv render context it frees
    // only on exit — alive while the main thread already sees an empty
    // slot and re-attaches, forcing the race window every time.
    let armed = Arc::new(AtomicBool::new(false));
    let detached = Arc::new(AtomicBool::new(false));
    {
        let cb_engine = Arc::clone(&engine);
        let armed = Arc::clone(&armed);
        let detached = Arc::clone(&detached);
        engine
            .set_render_update_callback(move || {
                if armed.load(Ordering::SeqCst) && !detached.swap(true, Ordering::SeqCst) {
                    cb_engine.detach_render();
                    std::thread::sleep(Duration::from_millis(500));
                }
            })
            .expect("replace callback");
    }
    armed.store(true, Ordering::SeqCst);
    engine.set_export_size(48, 48).expect("force a publish");

    let deadline = Instant::now() + Duration::from_secs(10);
    while engine.attached_render().is_some() {
        assert!(
            Instant::now() < deadline,
            "detach from on_update did not complete"
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    // Deliberately no sleep: the empty render slot already implies the
    // old thread's handle is parked, and the attach joins it — success
    // must be deterministic, not lucky timing.
    let (tx, rx) = std::sync::mpsc::channel();
    engine
        .attach_exported_render(mpv_engine::ExportOptions::new(64, 64), move || {
            let _ = tx.send(());
        })
        .expect("re-attach after a callback detach should succeed first try");

    // And the new backend is actually live: a forced render publishes.
    engine.set_export_size(40, 40).expect("resize");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(
            Instant::now() < deadline,
            "no frame from the re-attached backend"
        );
        let _ = rx.recv_timeout(Duration::from_millis(200));
        match engine.acquire_frame().expect("acquire") {
            Some(frame) if (frame.width(), frame.height()) == (40, 40) => break,
            _ => {}
        }
    }
    engine.detach_render();
}
