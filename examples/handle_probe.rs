//! Windows handle-growth probe: runs a load/stop loop and prints the
//! process's kernel-handle and thread counts every ten cycles, so growth
//! the harness detects can be *attributed* to a specific code path.
//!
//! ```sh
//! cargo run --example handle_probe -- headless 60
//! cargo run --features export --example handle_probe -- export 60
//! cargo run --features export --example handle_probe -- resize 40
//! cargo run --features wgpu   --example handle_probe -- import 40
//! ```
//!
//! Each mode adds exactly one thing to the one before it, so the mode
//! where the counter starts moving names the culprit:
//!
//! * `headless` — `vo=null`, no export backend, no window, no GPU. The
//!   control for mpv's own load/stop path.
//! * `export` — same loop with the exported-frame backend attached.
//! * `resize` — adds `set_export_size` churn, which retires the whole
//!   pool and reallocates it every cycle.
//! * `import` — adds `into_wgpu_texture`, which on Windows opens the
//!   buffer's shared handle on a consumer D3D12 device and *retires* the
//!   pool buffer, so a fresh shared texture is allocated per imported
//!   frame.
//!
//! Measured on Windows/NVIDIA, 40 cycles, five imported frames per
//! cycle:
//!
//! | mode | handles | verdict |
//! |---|---|---|
//! | `headless` | +0 | flat |
//! | `export` | +0 | flat |
//! | `resize` | +37 then flat | one-time, no growth |
//! | `import` | +284, ~1 per imported frame | **grows without bound** |
//!
//! The growth is not this crate's bookkeeping: instrumenting
//! `SurfaceBuffer` shows creates, drops and interop unregistrations all
//! balance exactly (`made=53 dropped=53 gl_freed=53`), so every NT
//! handle we create is closed and every interop object unregistered.
//! Nor is it wgpu holding resources: dropping the wgpu device returns
//! only a handful, while `detach_render` — which destroys the D3D11
//! device, the interop device and the GL context — returns *all* of
//! them.
//!
//! What is left is that opening a shared handle on a consumer device
//! leaves a reference on the producer's D3D11 device that lives until
//! that device is destroyed. That alone would be harmless; what turns it
//! into unbounded growth is the Windows import policy of retiring the
//! pool buffer per imported frame, so a long-running player allocates a
//! new shared texture — and strands a new reference — for every frame it
//! displays. `resize` proves buffer churn by itself is not the problem;
//! `import` isolates it to the open.

#[cfg(target_os = "windows")]
mod probe {
    use std::ffi::c_void;
    use std::path::Path;
    use std::time::{Duration, Instant};

    use mpv_engine::Engine;

    #[repr(C)]
    struct ThreadEntry32 {
        size: u32,
        cnt_usage: u32,
        thread_id: u32,
        owner_process_id: u32,
        base_pri: i32,
        delta_pri: i32,
        flags: u32,
    }

    const TH32CS_SNAPTHREAD: u32 = 0x0000_0004;
    const INVALID_HANDLE_VALUE: *mut c_void = usize::MAX as *mut c_void;

    const GR_GDIOBJECTS: u32 = 0;
    const GR_USEROBJECTS: u32 = 1;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetGuiResources(process: *mut c_void, flags: u32) -> u32;
    }

    /// GDI + USER objects: where a leaked window, DC or GL context shows
    /// up, as opposed to the kernel handles `handles()` counts.
    fn gui() -> u32 {
        unsafe {
            let process = GetCurrentProcess();
            GetGuiResources(process, GR_GDIOBJECTS) + GetGuiResources(process, GR_USEROBJECTS)
        }
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetProcessHandleCount(process: *mut c_void, count: *mut u32) -> i32;
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut c_void;
        fn Thread32First(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        fn Thread32Next(snapshot: *mut c_void, entry: *mut ThreadEntry32) -> i32;
        fn CloseHandle(object: *mut c_void) -> i32;
    }

    fn handles() -> u32 {
        let mut count = 0u32;
        unsafe {
            GetProcessHandleCount(GetCurrentProcess(), &mut count);
        }
        count
    }

    /// Live threads in this process. If handle growth tracks this, the
    /// leak is thread handles — a thread that exited without its handle
    /// being closed, or one that never exited at all.
    fn threads() -> u32 {
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snapshot == INVALID_HANDLE_VALUE || snapshot.is_null() {
                return 0;
            }
            let me = GetCurrentProcessId();
            let mut entry: ThreadEntry32 = std::mem::zeroed();
            entry.size = size_of::<ThreadEntry32>() as u32;
            let mut count = 0;
            let mut ok = Thread32First(snapshot, &mut entry);
            while ok != 0 {
                if entry.owner_process_id == me {
                    count += 1;
                }
                ok = Thread32Next(snapshot, &mut entry);
            }
            CloseHandle(snapshot);
            count
        }
    }

    fn generate_clip(target: &Path) -> bool {
        std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=s=320x240:r=30",
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
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Spin until `cond` or the deadline; returns whether it held.
    fn until(secs: f64, mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(secs);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    pub fn run() -> i32 {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mode = args.first().map(String::as_str).unwrap_or("headless");
        let cycles: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(60);

        let dir = tempfile::tempdir().expect("tempdir");
        let clip = dir.path().join("probe.nut");
        if !generate_clip(&clip) {
            eprintln!("skipping: ffmpeg unavailable");
            return 0;
        }
        let clip = clip.to_str().expect("utf-8 path").to_owned();

        let engine = match mode {
            "headless" => Engine::headless().property("ao", "null").build(),
            #[cfg(feature = "export")]
            "export" | "resize" | "attach" => Engine::video().property("ao", "null").build(),
            #[cfg(feature = "wgpu")]
            "import" => Engine::video().property("ao", "null").build(),
            other => {
                eprintln!("unknown mode {other:?} (try: headless, export)");
                return 2;
            }
        }
        .expect("engine");

        #[cfg(feature = "export")]
        if matches!(mode, "export" | "resize" | "import") {
            engine
                .attach_exported_render(mpv_engine::ExportOptions::new(320, 240), || {})
                .expect("attach exported render");
        }

        println!("mode={mode} cycles={cycles}");
        println!(
            "{:>6} {:>9} {:>9} {:>7}",
            "cycle", "handles", "threads", "gui"
        );
        // Warm-up outside the measurement: the first load pulls in
        // demuxers, codecs and their threads.
        #[cfg(feature = "export")]
        if mode == "attach" {
            engine
                .attach_exported_render(mpv_engine::ExportOptions::new(320, 240), || {})
                .expect("attach");
        }
        for _ in 0..3 {
            let _ = engine.load(&clip);
            until(10.0, || !engine.is_idle());
            let _ = engine.stop();
            until(10.0, || engine.is_idle());
        }
        std::thread::sleep(Duration::from_millis(500));
        let (h0, t0, g0) = (handles(), threads(), gui());
        println!("{:>6} {h0:>9} {t0:>9} {g0:>7}   <- baseline", 0);

        // Only built for the "import" mode; a device with no surface is
        // all the frame import needs.
        #[cfg(feature = "wgpu")]
        let device = (mode == "import").then(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = mpv_engine::WGPU_BACKEND;
            let instance = wgpu::Instance::new(descriptor);
            let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
                .expect("wgpu adapter");
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: mpv_engine::REQUIRED_WGPU_FEATURES,
                ..Default::default()
            }))
            .expect("wgpu device")
        });

        for i in 1..=cycles {
            if engine.load(&clip).is_err() || !until(10.0, || !engine.is_idle()) {
                eprintln!("cycle {i}: load did not start");
                return 1;
            }
            // Per-mode extra work, matching what the harness's load-churn
            // cycle does beyond a bare load/stop.
            #[cfg(feature = "export")]
            if mode == "attach" {
                // The GL context, its hidden window and DC, the D3D11
                // device and the render thread are all built and torn
                // down here — where a leaked USER/GDI object would land.
                engine.detach_render();
                engine
                    .attach_exported_render(mpv_engine::ExportOptions::new(320, 240), || {})
                    .expect("re-attach");
            }
            #[cfg(feature = "export")]
            if mode == "resize" {
                for (w, h) in [(480u32, 270u32), (320, 240)] {
                    let _ = engine.set_export_size(w, h);
                    until(10.0, || {
                        matches!(engine.acquire_frame(), Ok(Some(f))
                            if (f.width(), f.height()) == (w, h))
                    });
                }
            }
            #[cfg(feature = "wgpu")]
            if let Some((device, queue)) = device.as_ref() {
                // Import and drop several frames: on Windows each import
                // opens the shared handle on the consumer device and
                // *retires* the pool buffer, so the pool allocates a
                // replacement every time. That is the path the harness
                // exercises and this bare loop did not.
                let mut imported = 0;
                until(10.0, || {
                    if let Ok(Some(frame)) = engine.acquire_frame() {
                        match frame.into_wgpu_texture(device) {
                            Ok(texture) => {
                                drop(texture);
                                imported += 1;
                            }
                            Err(e) => eprintln!("import failed: {e}"),
                        }
                    }
                    imported >= 5
                });
                // wgpu frees a dropped resource only once the queue
                // fence passes the point where it was last used, so a
                // device that never submits anything never advances and
                // never frees. `MPV_PROBE_SUBMIT=1` adds an empty submit
                // to distinguish "wgpu is holding it" from "it leaked".
                if std::env::var_os("MPV_PROBE_SUBMIT").is_some() {
                    let encoder = device.create_command_encoder(&Default::default());
                    queue.submit([encoder.finish()]);
                }
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
            if engine.stop().is_err() || !until(10.0, || engine.is_idle()) {
                eprintln!("cycle {i}: stop did not settle");
                return 1;
            }
            if i % 10 == 0 || i == cycles {
                // Let exiting threads and deferred frees land, or the
                // sample reads low for reasons that have nothing to do
                // with a leak.
                std::thread::sleep(Duration::from_millis(800));
                let (h, t, g) = (handles(), threads(), gui());
                println!(
                    "{i:>6} {h:>9} {t:>9} {g:>7}   ({:+} handles, {:+} threads, {:+} gui; \
                     {:.2} handles/cycle, {:.2} gui/cycle)",
                    h as i64 - h0 as i64,
                    t as i64 - t0 as i64,
                    g as i64 - g0 as i64,
                    (h as f64 - h0 as f64) / i as f64,
                    (g as f64 - g0 as f64) / i as f64,
                );
            }
        }
        // Does anything ever give the handles back? Tear the consumer
        // down and look again: if they return here, they were held by
        // wgpu/D3D12 rather than leaked outright.
        #[cfg(feature = "wgpu")]
        if device.is_some() {
            drop(device);
            std::thread::sleep(Duration::from_millis(1500));
            println!(
                "after dropping the wgpu device: {} handles ({:+} vs baseline)",
                handles(),
                handles() as i64 - h0 as i64
            );
        }
        #[cfg(feature = "export")]
        engine.detach_render();
        std::thread::sleep(Duration::from_millis(1000));
        println!(
            "after detaching the engine:      {} handles ({:+} vs baseline)",
            handles(),
            handles() as i64 - h0 as i64
        );
        0
    }
}

#[cfg(target_os = "windows")]
fn main() {
    std::process::exit(probe::run());
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("this probe reads Windows process counters");
}
