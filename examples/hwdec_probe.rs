//! Does zero-copy VA-API engage on the exported-frame backend?
//!
//! Builds an engine with mpv's own hwdec/vaapi logging on, attaches the
//! exported render backend (the surfaceless EGL/GBM context), plays a file
//! for a few seconds, and reports the decoder + interop mpv actually
//! selected plus the effective render rate.
//!
//!   cargo run --features export --example hwdec_probe -- <file> [hwdec]
//!
//! `hwdec` defaults to `auto-safe` (what `Engine::video()` uses); pass
//! `vaapi` to force the zero-copy GL-interop path, `vaapi-copy` for the
//! readback path, or `no` for software.

#[cfg(export_backend)]
fn main() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use mpv_engine::{Engine, ExportOptions};

    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: hwdec_probe <file> [hwdec] [export_w export_h]");
        std::process::exit(2);
    };
    let hwdec = args.next().unwrap_or_else(|| "auto-safe".into());
    let ew: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1920);
    let eh: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1080);

    let engine = Engine::builder()
        .property("vo", "libmpv")
        .property("keep-open", "always")
        .property("hwdec", &hwdec)
        // mpv logs to stderr with terminal=yes; crank the subsystems that
        // explain a hwdec decision, keep the rest quiet.
        .property("terminal", "yes")
        .property("input-terminal", "no")
        .property("quiet", "yes")
        .property(
            "msg-level",
            &std::env::var("MPV_MSG_LEVEL").unwrap_or_else(|_| {
                "all=error,vd=v,vd-lavc=v,vaapi=v,ffmpeg=v,gpu=v,vo=v,hwdec=trace".into()
            }),
        )
        .build()
        .expect("build engine");

    let frames = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&frames);
    engine
        .attach_exported_render(ExportOptions::new(ew, eh), move || {
            counter.fetch_add(1, Ordering::Relaxed);
        })
        .expect("attach exported render");
    engine.load_when_ready(&path).expect("load");

    eprintln!("\n### probe: file={path} hwdec-requested={hwdec} export={ew}x{eh}\n");

    let start = Instant::now();
    let run = Duration::from_secs(
        std::env::var("RUN_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8),
    );
    let mut probed = false;
    let mut probe_at = Duration::from_secs(3);
    let mut frames_at_probe = 0u64;
    let mut t_at_probe = start;
    while start.elapsed() < run {
        for _ev in engine.pump_events() {}
        // Drain and drop frames so the pool keeps cycling like a real shell.
        while let Ok(Some(_frame)) = engine.acquire_frame() {}
        std::thread::sleep(Duration::from_millis(4));
        if !probed && start.elapsed() >= probe_at {
            probed = true;
            frames_at_probe = frames.load(Ordering::Relaxed);
            t_at_probe = Instant::now();
            probe_at = Duration::MAX;
        }
    }

    // Effective render rate over the post-warmup window.
    let df = frames
        .load(Ordering::Relaxed)
        .saturating_sub(frames_at_probe);
    let dt = t_at_probe.elapsed().as_secs_f64();
    eprintln!("\n### RESULT for hwdec={hwdec} export={ew}x{eh}");
    let g = |p: &str| -> String {
        engine
            .get_property::<String>(p)
            .unwrap_or_else(|_| "<err>".into())
    };
    for p in [
        "hwdec-current",
        "hwdec-interop",
        "video-codec",
        "video-params/w",
        "video-params/h",
        "container-fps",
        "estimated-vf-fps",
        "decoder-frame-drop-count",
        "frame-drop-count",
    ] {
        eprintln!("  {p:28} = {}", g(p));
    }
    eprintln!(
        "  {:28} = {:.1} fps ({df} frames / {dt:.2}s exported)",
        "effective-export-rate",
        df as f64 / dt
    );
    engine.detach_render();
}

#[cfg(not(export_backend))]
fn main() {
    eprintln!("hwdec_probe needs --features export on macOS, Linux, or Windows");
}
