//! Real-window test harness for the exported-frame backend: drives a
//! live winit window through the lifecycle scenarios the integration
//! suites can't reach, because every one of them needs a real window, a
//! real swapchain and a real compositor.
//!
//! ```sh
//! # scripted: runs every scenario, asserts, exits non-zero on failure
//! cargo run --features wgpu --example harness
//!
//! # one scenario by name (substring match)
//! cargo run --features wgpu --example harness -- resize
//! cargo run --features wgpu --example harness -- leaks
//!
//! # drive it by hand and watch
//! cargo run --features wgpu --example harness -- --interactive
//!
//! # longer soak for the leak scenarios (default 8 cycles each)
//! MPV_HARNESS_LEAK_CYCLES=40 cargo run --features wgpu --example harness -- leaks
//! ```
//!
//! Scenarios: `resize`, `fullscreen`, `play/pause/stop/start`,
//! `teardown-while-playing`, `back-to-back loads`, `multi-engine one
//! window`, and three `resource leaks` variants.
//!
//! The three leak variants exist to *attribute* growth rather than just
//! detect it, by changing exactly one thing about what each cycle tears
//! down:
//!
//! * **load churn** — one engine, attached once, never detached; only
//!   the file is reloaded. The control for mpv accumulating on a
//!   long-lived core.
//! * **attach churn** — one engine, but the exported backend is
//!   attached and detached every cycle. This is the code this crate
//!   owns: GL context, buffer pool, platform handles, render thread.
//! * **engine churn** — a whole new `Engine` every cycle, mpv core
//!   included.
//!
//! Growth that shows in one variant but not the others says where it
//! lives. Kernel handles, file descriptors and GDI/USER objects are
//! budgeted as **totals for the run, not per cycle**: every one of them
//! is closed by an explicit `Drop`, so the correct answer is zero
//! however long the run is. Memory is judged on trend instead (early
//! cycles vs late), because allocator arenas and driver caches
//! legitimately grow and never give it back.
//!
//! Resource sampling is the one genuinely per-platform piece (see
//! [`sample`]), because what leaks and how you count it differs:
//!
//! | | Windows | Linux | macOS |
//! |---|---|---|---|
//! | kernel handles | `GetProcessHandleCount` | — | Mach port names |
//! | file descriptors | — | `/proc/self/fd` | `PROC_PIDLISTFDS` |
//! | GDI + USER objects | `GetGuiResources` | n/a | n/a |
//! | threads | — | `/proc/self/status` | `PROC_PIDTASKINFO` |
//! | memory | `PrivateUsage` | `VmRSS` | `ri_phys_footprint` |
//!
//! On a platform with no sampler the leak scenarios report themselves
//! unmeasured rather than passing vacuously — and even where there is
//! one, a sensitivity probe first proves the counters actually move when
//! an engine exists, so a sampler that silently reads nothing fails
//! instead of reporting a perfect zero forever.
//!
//! Clips are synthesized with ffmpeg at startup (distinct pattern *and*
//! distinct duration per clip, so "which video is on screen" is
//! assertable through `duration`), into a tempdir deleted on exit.
//! Without ffmpeg the harness skips rather than fails.

#[cfg(all(
    feature = "wgpu",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod harness {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use mpv_engine::{Engine, ExportOptions, ExportedFrame, PlaybackEvent};
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::{ElementState, KeyEvent, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Fullscreen, Window, WindowId};

    /// Windowed size the harness returns to between scenarios.
    const BASE_SIZE: (u32, u32) = (960, 540);
    /// Scenario ticks per second — fast enough that a resize settles
    /// promptly, slow enough not to spin a core.
    const TICK: Duration = Duration::from_millis(16);
    /// `ExportOptions`' default pool, mirrored here because the
    /// teardown scenario deliberately exhausts it.
    const POOL_SIZE: usize = 3;

    // ---- clips -------------------------------------------------------

    /// One synthesized clip. `secs` doubles as the clip's identity: the
    /// scenarios assert on `Engine::duration` to prove *which* file is
    /// loaded, which a pixel check could only do approximately.
    struct Clip {
        label: &'static str,
        path: PathBuf,
        secs: f64,
    }

    /// lavfi source, duration, human label. Durations are deliberately
    /// distinct and integral.
    const CLIP_SPECS: [(&str, &str, f64); 3] = [
        ("A red", "color=c=red:s=320x240:r=30", 2.0),
        ("B testsrc", "testsrc=s=320x240:r=60", 3.0),
        ("C smptebars", "smptebars=s=640x360:r=30", 4.0),
    ];

    /// Synthesize the clips into `dir` (rawvideo in NUT — no encoder
    /// needed, same trick as the integration suites). `None` when
    /// ffmpeg is missing or fails, which the caller turns into a skip.
    fn generate_clips(dir: &Path) -> Option<Vec<Clip>> {
        let mut clips = Vec::new();
        for (label, source, secs) in CLIP_SPECS {
            let path = dir.join(format!("{}.nut", label.split(' ').next().unwrap_or(label)));
            let status = std::process::Command::new("ffmpeg")
                .args(["-y", "-f", "lavfi", "-i", source, "-t"])
                .arg(secs.to_string())
                .args(["-c:v", "rawvideo", "-pix_fmt", "yuv420p"])
                .arg(&path)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            match status {
                Ok(s) if s.success() => clips.push(Clip { label, path, secs }),
                _ => {
                    eprintln!("skipping: ffmpeg unavailable or failed on '{source}'");
                    return None;
                }
            }
        }
        Some(clips)
    }

    // ---- resource sampling -------------------------------------------

    /// A snapshot of what this process is holding. Every field is
    /// optional because *what* leaks — and how you count it — is
    /// per-OS: kernel handles and GDI/USER objects on Windows (the
    /// export tier's shared NT handles, D3D11 objects, the hidden
    /// window, its DC and its GL context all land there), open file
    /// descriptors on Linux (one dmabuf fd per pool buffer), and
    /// resident memory everywhere. The churn scenario and its
    /// thresholds are common; only [`sample`] is platform code.
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    struct Resources {
        /// Kernel object references: open handles on Windows, Mach port
        /// names on macOS — where every IOSurface, CGL context, Metal
        /// object, thread and semaphore this process holds is a port
        /// right, so a leaked pool buffer or GL context lands here even
        /// when memory noise hides it.
        handles: Option<u64>,
        /// Open file descriptors (Linux and macOS).
        fds: Option<u64>,
        /// GDI + USER objects (Windows only).
        gui: Option<u64>,
        /// OS threads in the process (Linux and macOS). The export
        /// backend owns a render thread per attach; a leaked one (the
        /// class of bug the orphan-parking fix in `detach_render`
        /// closed) is invisible to the handle and memory counters at
        /// this scale but unmistakable here.
        threads: Option<u64>,
        /// Bytes the process has committed for itself: `PrivateUsage`
        /// on Windows, `VmRSS` on Linux, `ri_phys_footprint` on macOS.
        /// Deliberately *not* Windows' working-set size — the OS trims
        /// and refills that on its own schedule, so it drifts by
        /// megabytes for reasons that have nothing to do with this
        /// process. On macOS, footprint rather than resident size
        /// because the kernel attributes IOSurface memory — the pool's
        /// entire currency — to footprint, while resident size can miss
        /// it.
        rss: Option<u64>,
    }

    impl Resources {
        /// Field-wise `self - earlier`; a field absent from either side
        /// stays absent. Signed on purpose: counters go *down* as well
        /// as up, and clamping a decrease to zero makes a fall-then-rise
        /// read as steady growth — which is exactly how a warm-up
        /// artifact gets mistaken for a leak.
        fn since(self, earlier: Self) -> Deltas {
            let d = |now: Option<u64>, was: Option<u64>| Some(now? as i64 - was? as i64);
            Deltas {
                handles: d(self.handles, earlier.handles),
                fds: d(self.fds, earlier.fds),
                gui: d(self.gui, earlier.gui),
                threads: d(self.threads, earlier.threads),
                rss: d(self.rss, earlier.rss),
            }
        }

        fn is_empty(self) -> bool {
            self == Self::default()
        }
    }

    #[derive(Clone, Copy)]
    struct Deltas {
        handles: Option<i64>,
        fds: Option<i64>,
        gui: Option<i64>,
        threads: Option<i64>,
        rss: Option<i64>,
    }

    impl std::fmt::Display for Deltas {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut parts = Vec::new();
            if let Some(v) = self.handles {
                parts.push(format!("handles {v:+}"));
            }
            if let Some(v) = self.fds {
                parts.push(format!("fds {v:+}"));
            }
            if let Some(v) = self.gui {
                parts.push(format!("gui {v:+}"));
            }
            if let Some(v) = self.threads {
                parts.push(format!("threads {v:+}"));
            }
            if let Some(v) = self.rss {
                parts.push(format!("mem {:+.1}MiB", v as f64 / (1024.0 * 1024.0)));
            }
            f.write_str(&parts.join(", "))
        }
    }

    #[cfg(target_os = "windows")]
    fn sample() -> Resources {
        use std::ffi::c_void;

        /// `PROCESS_MEMORY_COUNTERS_EX` — the `_EX` tail is
        /// `private_usage`, which is the committed-private figure we
        /// actually want.
        #[repr(C)]
        struct ProcessMemoryCountersEx {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
            private_usage: usize,
        }
        /// `GR_GDIOBJECTS` / `GR_USEROBJECTS`.
        const GR_GDIOBJECTS: u32 = 0;
        const GR_USEROBJECTS: u32 = 1;

        // Declared directly rather than pulled from a crate, matching
        // `src/export/windows.rs`: `K32GetProcessMemoryInfo` is the
        // kernel32 forwarder for the psapi entry point, so kernel32 and
        // user32 cover all four.
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn GetProcessHandleCount(process: *mut c_void, count: *mut u32) -> i32;
            fn K32GetProcessMemoryInfo(
                process: *mut c_void,
                counters: *mut ProcessMemoryCountersEx,
                cb: u32,
            ) -> i32;
        }
        #[link(name = "user32")]
        unsafe extern "system" {
            fn GetGuiResources(process: *mut c_void, flags: u32) -> u32;
        }

        unsafe {
            let process = GetCurrentProcess();
            let mut count = 0u32;
            let handles = (GetProcessHandleCount(process, &mut count) != 0).then_some(count as u64);
            let mut counters: ProcessMemoryCountersEx = std::mem::zeroed();
            counters.cb = size_of::<ProcessMemoryCountersEx>() as u32;
            let rss = (K32GetProcessMemoryInfo(process, &mut counters, counters.cb) != 0)
                .then_some(counters.private_usage as u64);
            let gui = Some(u64::from(
                GetGuiResources(process, GR_GDIOBJECTS) + GetGuiResources(process, GR_USEROBJECTS),
            ));
            Resources {
                handles,
                fds: None,
                gui,
                // No one-call thread count on Win32 (a Toolhelp snapshot
                // walks every process); handles subsume threads there.
                threads: None,
                rss,
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn sample() -> Resources {
        // One dmabuf fd per pool buffer, so the fd count is the sharpest
        // signal here; `VmRSS` catches the rest.
        let fds = std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|entries| entries.count() as u64);
        let status = std::fs::read_to_string("/proc/self/status").ok();
        let field = |prefix: &str| {
            status.as_deref()?.lines().find_map(|line| {
                line.strip_prefix(prefix)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
        };
        let rss = field("VmRSS:").map(|kb| kb * 1024);
        let threads = field("Threads:");
        Resources {
            handles: None,
            fds,
            gui: None,
            threads,
            rss,
        }
    }

    /// The macOS sampler leans on two seams, declared directly in the
    /// style of `src/export/macos.rs` rather than through a crate:
    /// libproc (`proc_pidinfo` / `proc_pid_rusage` — what Activity
    /// Monitor itself is built on) and Mach (`mach_port_names`). Both
    /// live in libSystem, so nothing new is linked.
    ///
    /// Mach port names are the macOS analog of the Windows handle
    /// count, and the sharpest counter for *this* crate's leak surface:
    /// an IOSurface, a CGL context, a Metal object, a thread and a
    /// semaphore are all port rights in this table, so a pool buffer or
    /// hidden GL context that outlives its teardown shows up as +1 here
    /// even when it is far below the memory noise floor. Memory itself
    /// is `ri_phys_footprint` rather than resident size because the
    /// kernel attributes IOSurface pages — the pool's entire currency —
    /// to footprint, while resident size can miss purgeable/nonvolatile
    /// surface memory entirely.
    #[cfg(target_os = "macos")]
    fn sample() -> Resources {
        use std::ffi::{c_int, c_void};

        const PROC_PIDLISTFDS: c_int = 1;
        const PROC_PIDTASKINFO: c_int = 4;
        /// `struct proc_fdinfo`: `i32` fd + `u32` fdtype.
        const PROC_PIDLISTFD_SIZE: usize = 8;
        const RUSAGE_INFO_V0: c_int = 0;

        /// `struct proc_taskinfo` (sys/proc_info.h).
        #[repr(C)]
        struct ProcTaskInfo {
            pti_virtual_size: u64,
            pti_resident_size: u64,
            pti_total_user: u64,
            pti_total_system: u64,
            pti_threads_user: u64,
            pti_threads_system: u64,
            pti_policy: i32,
            pti_faults: i32,
            pti_pageins: i32,
            pti_cow_faults: i32,
            pti_messages_sent: i32,
            pti_messages_received: i32,
            pti_syscalls_mach: i32,
            pti_syscalls_unix: i32,
            pti_csw: i32,
            pti_threadnum: i32,
            pti_numrunning: i32,
            pti_priority: i32,
        }

        /// `struct rusage_info_v0` (sys/resource.h) — the v0 revision
        /// already carries `ri_phys_footprint`, so the longer ones are
        /// not needed.
        #[repr(C)]
        struct RusageInfoV0 {
            ri_uuid: [u8; 16],
            ri_user_time: u64,
            ri_system_time: u64,
            ri_pkg_idle_wkups: u64,
            ri_interrupt_wkups: u64,
            ri_pageins: u64,
            ri_wired_size: u64,
            ri_resident_size: u64,
            ri_phys_footprint: u64,
            ri_proc_start_abstime: u64,
            ri_proc_exit_abstime: u64,
        }

        unsafe extern "C" {
            fn proc_pidinfo(
                pid: c_int,
                flavor: c_int,
                arg: u64,
                buffer: *mut c_void,
                buffersize: c_int,
            ) -> c_int;
            fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut RusageInfoV0) -> c_int;
            /// `mach_task_self()` is a C macro over this global.
            static mach_task_self_: u32;
            fn mach_port_names(
                task: u32,
                names: *mut *mut u32,
                names_count: *mut u32,
                types: *mut *mut u32,
                types_count: *mut u32,
            ) -> i32;
            fn vm_deallocate(task: u32, address: usize, size: usize) -> i32;
        }

        let pid = std::process::id() as c_int;

        // Port-name table size. The two arrays come back vm_allocate'd
        // in our own address space and are handed straight back — only
        // the count matters.
        let handles = unsafe {
            let task = mach_task_self_;
            let mut names: *mut u32 = std::ptr::null_mut();
            let mut names_count: u32 = 0;
            let mut types: *mut u32 = std::ptr::null_mut();
            let mut types_count: u32 = 0;
            (mach_port_names(
                task,
                &mut names,
                &mut names_count,
                &mut types,
                &mut types_count,
            ) == 0)
                .then(|| {
                    if !names.is_null() {
                        vm_deallocate(task, names as usize, names_count as usize * 4);
                    }
                    if !types.is_null() {
                        vm_deallocate(task, types as usize, types_count as usize * 4);
                    }
                    u64::from(names_count)
                })
        };

        // Fd count: a null-buffer call returns the kernel's byte-size
        // estimate (deliberately padded), so the real read follows with
        // headroom — the count comes from the bytes actually written,
        // which the padding does not inflate.
        let fds = unsafe {
            let hint = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, std::ptr::null_mut(), 0);
            if hint <= 0 {
                None
            } else {
                let capacity = hint as usize * 2 + 64 * PROC_PIDLISTFD_SIZE;
                let mut buffer = vec![0u8; capacity];
                let written = proc_pidinfo(
                    pid,
                    PROC_PIDLISTFDS,
                    0,
                    buffer.as_mut_ptr().cast(),
                    capacity as c_int,
                );
                (written > 0).then(|| written as u64 / PROC_PIDLISTFD_SIZE as u64)
            }
        };

        let threads = unsafe {
            let mut info: ProcTaskInfo = std::mem::zeroed();
            let size = size_of::<ProcTaskInfo>() as c_int;
            let written = proc_pidinfo(
                pid,
                PROC_PIDTASKINFO,
                0,
                (&mut info as *mut ProcTaskInfo).cast(),
                size,
            );
            (written == size).then_some(info.pti_threadnum as u64)
        };

        let rss = unsafe {
            let mut info: RusageInfoV0 = std::mem::zeroed();
            (proc_pid_rusage(pid, RUSAGE_INFO_V0, &mut info) == 0).then_some(info.ri_phys_footprint)
        };

        Resources {
            handles,
            fds,
            gui: None,
            threads,
            rss,
        }
    }

    // ---- gpu ---------------------------------------------------------

    struct Gfx {
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
    }

    fn gfx_for_window(window: Arc<Window>) -> Gfx {
        let mut descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()));
        descriptor.backends = mpv_engine::WGPU_BACKEND;
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no wgpu adapter for the native backend");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: mpv_engine::REQUIRED_WGPU_FEATURES,
            ..Default::default()
        }))
        .expect("wgpu device");

        let caps = surface.get_capabilities(&adapter);
        assert!(
            caps.formats.contains(&wgpu::TextureFormat::Bgra8Unorm),
            "surface does not offer Bgra8Unorm (got {:?})",
            caps.formats
        );
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        Gfx {
            surface,
            device,
            queue,
            config,
        }
    }

    // ---- panes -------------------------------------------------------

    /// A destination rectangle in the window, in physical pixels.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Rect {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    }

    /// One engine rendering into one rectangle of the window. Several
    /// panes is the "multiple videos in the same window" case: separate
    /// mpv cores, separate export backends, separate pools, one
    /// swapchain.
    struct Pane {
        engine: Engine,
        rect: Rect,
        /// Newest imported frame, retained so a repaint with no new
        /// video (paused, ended, or just a resize) still has something
        /// to present.
        last: Option<wgpu::Texture>,
        /// Size of `last`, for the resize/fullscreen assertions.
        last_size: Option<(u32, u32)>,
        /// Frames imported as textures.
        frames: u64,
        /// Frames acquired, whether imported or held.
        acquires: u64,
        /// While > 0, acquired frames are parked in `held` instead of
        /// imported — this is how the teardown scenario exhausts the
        /// pool on purpose.
        hold_target: usize,
        held: Vec<ExportedFrame>,
        /// Acquires that succeeded *after* the pool should have been
        /// exhausted. Any of these is a pool-accounting bug.
        over_acquires: u64,
    }

    impl Pane {
        fn new(engine: Engine, rect: Rect) -> Self {
            Self {
                engine,
                rect,
                last: None,
                last_size: None,
                frames: 0,
                acquires: 0,
                hold_target: 0,
                held: Vec::new(),
                over_acquires: 0,
            }
        }
    }

    // ---- harness -----------------------------------------------------

    struct Harness {
        window: Arc<Window>,
        gfx: Gfx,
        clips: Vec<Clip>,
        panes: Vec<Pane>,
        proxy: EventLoopProxy<()>,
        /// Set by any pane's `Failed` playback event; scenarios fail on
        /// it rather than hanging until their deadline.
        playback_error: Option<String>,
        /// Observations a scenario wants in its result line — things
        /// worth *recording* rather than asserting, like which way mpv
        /// reacted to a teardown.
        notes: Vec<String>,
        /// The window size `relayout` last applied. A `Resized` event is
        /// supposed to drive relayout, but Wayland compositors coalesce or
        /// drop those — a window can reach a new size with no event ever
        /// delivered, stranding the panes at the old export size. The
        /// runner reconciles against this each tick so a missed event
        /// can't wedge a resize scenario (see `reconcile_layout`).
        layout_size: (u32, u32),
    }

    impl Harness {
        fn window_size(&self) -> (u32, u32) {
            let s = self.window.inner_size();
            (s.width.max(1), s.height.max(1))
        }

        /// Tile `n` panes across the window, left to right.
        fn layout(&self, n: usize) -> Vec<Rect> {
            let (w, h) = self.window_size();
            let each = (w / n.max(1) as u32).max(1);
            (0..n)
                .map(|i| Rect {
                    x: each * i as u32,
                    y: 0,
                    // Last pane takes the rounding remainder.
                    w: if i + 1 == n {
                        w - each * i as u32
                    } else {
                        each
                    },
                    h,
                })
                .collect()
        }

        /// Tear every pane down and build `n` fresh ones, each with its
        /// own engine and exported backend attached. Panes are dropped
        /// first so the old engines' render threads are joined before
        /// new ones start.
        fn set_panes(&mut self, n: usize) -> Result<(), String> {
            for pane in self.panes.drain(..) {
                pane.engine.detach_render();
            }
            self.playback_error = None;
            for rect in self.layout(n) {
                let engine = Engine::video()
                    .property("ao", "null")
                    .build()
                    .map_err(|e| format!("engine build: {e}"))?;
                let on_update = {
                    let proxy = self.proxy.clone();
                    move || {
                        let _ = proxy.send_event(());
                    }
                };
                engine
                    .attach_exported_render(ExportOptions::new(rect.w, rect.h), on_update)
                    .map_err(|e| format!("attach: {e}"))?;
                let proxy = self.proxy.clone();
                engine.set_wakeup_callback(move || {
                    let _ = proxy.send_event(());
                });
                self.panes.push(Pane::new(engine, rect));
            }
            Ok(())
        }

        /// Load `clip` into pane `i`, optionally looping so the scenario
        /// has an endless source. Uses `load`, not `load_when_ready`:
        /// the backend is already attached, and back-to-back loads must
        /// not queue behind anything.
        fn load(&mut self, i: usize, clip: usize, looping: bool) -> Result<(), String> {
            let path = self.clips[clip]
                .path
                .to_str()
                .ok_or("non-utf-8 clip path")?
                .to_owned();
            let pane = &mut self.panes[i];
            pane.engine
                .set_property("loop-file", if looping { "inf" } else { "no" })
                .map_err(|e| format!("loop-file: {e}"))?;
            pane.engine.load(&path).map_err(|e| format!("load: {e}"))?;
            Ok(())
        }

        /// Re-apply the current window size to the surface and to every
        /// pane's export size. Called on every `Resized`, and by
        /// [`reconcile_layout`] when an event was missed.
        fn relayout(&mut self) {
            let (w, h) = self.window_size();
            self.gfx.config.width = w;
            self.gfx.config.height = h;
            self.gfx
                .surface
                .configure(&self.gfx.device, &self.gfx.config);
            let rects = self.layout(self.panes.len());
            for (pane, rect) in self.panes.iter_mut().zip(rects) {
                pane.rect = rect;
                // Zero would pause rendering; `layout` already floors at 1.
                let _ = pane.engine.set_export_size(rect.w, rect.h);
            }
            self.layout_size = (w, h);
        }

        /// Relayout if the live window size has drifted from what
        /// `relayout` last applied — the safety net for `Resized` events a
        /// Wayland compositor coalesced or never sent. A no-op (no
        /// reconfigure, no `set_export_size`) whenever they already agree,
        /// so it is cheap to call every tick.
        fn reconcile_layout(&mut self) {
            if self.window_size() != self.layout_size {
                self.relayout();
                self.window.request_redraw();
            }
        }

        /// One-line-per-pane state dump, appended to any timeout so a
        /// failure says what the engine was actually doing rather than
        /// only that it stopped.
        fn diagnose(&self) -> String {
            let mut out = format!("window {:?}", self.window_size());
            if let Some(e) = &self.playback_error {
                out.push_str(&format!("; playback error: {e}"));
            }
            let round = |v: Option<f64>| v.map(|x| (x * 100.0).round() / 100.0);
            for (i, pane) in self.panes.iter().enumerate() {
                out.push_str(&format!(
                    "\n        pane {i}: render={:?} idle={} paused={} pos={:?} dur={:?} \
                     frames={} acquires={} held={} size={:?} rect={}x{}",
                    pane.engine.attached_render(),
                    pane.engine.is_idle(),
                    pane.engine.is_paused(),
                    round(pane.engine.position()),
                    round(pane.engine.duration()),
                    pane.frames,
                    pane.acquires,
                    pane.held.len(),
                    pane.last_size,
                    pane.rect.w,
                    pane.rect.h,
                ));
            }
            out
        }

        fn set_fullscreen(&self, on: bool) {
            self.window
                .set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
        }

        /// Drain each engine's event queue, recording the first failure.
        fn note(&mut self, note: impl Into<String>) {
            self.notes.push(note.into());
        }

        fn pump(&mut self) {
            for pane in &mut self.panes {
                for event in pane.engine.pump_events() {
                    match event {
                        PlaybackEvent::Failed { message, .. } => {
                            self.playback_error.get_or_insert(message);
                        }
                        PlaybackEvent::Shutdown => {
                            self.playback_error
                                .get_or_insert_with(|| "mpv core shut down".into());
                        }
                        _ => {}
                    }
                }
            }
        }

        /// Acquire from every pane, then composite: clear the surface
        /// and blit each pane's newest frame into its rectangle.
        fn redraw(&mut self) {
            for pane in &mut self.panes {
                let frame = match pane.engine.acquire_frame() {
                    Ok(Some(frame)) => frame,
                    Ok(None) => continue,
                    Err(_) => continue,
                };
                pane.acquires += 1;
                if pane.hold_target > 0 {
                    // Park it. Past the target this is a pool-accounting
                    // bug, but hold it anyway so the pool stays
                    // exhausted for the rest of the check.
                    if pane.held.len() >= pane.hold_target {
                        pane.over_acquires += 1;
                    }
                    pane.held.push(frame);
                    continue;
                }
                match frame.into_wgpu_texture(&self.gfx.device) {
                    Ok(texture) => {
                        pane.last_size = Some((texture.width(), texture.height()));
                        pane.last = Some(texture);
                        pane.frames += 1;
                    }
                    Err(e) => eprintln!("frame import failed: {e}"),
                }
            }
            if self.panes.iter().all(|p| p.last.is_none()) {
                // Never acquire a swapchain image we won't present: an
                // image dropped without `present` is not returned to the
                // presentation engine, and on Vulkan a few of those
                // starve the swapchain into permanent timeouts.
                return;
            }
            use wgpu::CurrentSurfaceTexture as Cst;
            let target = match self.gfx.surface.get_current_texture() {
                Cst::Success(target) | Cst::Suboptimal(target) => target,
                Cst::Timeout | Cst::Occluded => return,
                Cst::Outdated | Cst::Lost | Cst::Validation => {
                    self.gfx
                        .surface
                        .configure(&self.gfx.device, &self.gfx.config);
                    self.window.request_redraw();
                    return;
                }
            };
            let view = target.texture.create_view(&Default::default());
            let mut encoder = self.gfx.device.create_command_encoder(&Default::default());
            // Clear first: with several panes (or a pane smaller than
            // its rect mid-resize) the uncovered pixels are otherwise
            // whatever the swapchain image last held.
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let (sw, sh) = (target.texture.width(), target.texture.height());
            for pane in &self.panes {
                let Some(texture) = pane.last.as_ref() else {
                    continue;
                };
                // A frame can be a resize behind its rect (or the rect a
                // resize behind the surface); copy the intersection so a
                // stale size is shown cropped rather than dropped.
                let w = texture
                    .width()
                    .min(pane.rect.w)
                    .min(sw.saturating_sub(pane.rect.x));
                let h = texture
                    .height()
                    .min(pane.rect.h)
                    .min(sh.saturating_sub(pane.rect.y));
                if w == 0 || h == 0 {
                    continue;
                }
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    wgpu::TexelCopyTextureInfo {
                        texture: &target.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: pane.rect.x,
                            y: pane.rect.y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.gfx.queue.submit([encoder.finish()]);
            self.window.pre_present_notify();
            self.gfx.queue.present(target);
        }
    }

    // ---- scenario scripting -----------------------------------------

    type Enter = Box<dyn FnMut(&mut Harness) -> Result<(), String>>;
    type Poll = Box<dyn FnMut(&mut Harness) -> Result<bool, String>>;

    /// One step of a scenario: `enter` runs once, then `poll` runs every
    /// tick until it returns `true` or `timeout` elapses.
    struct Step {
        label: &'static str,
        enter: Enter,
        poll: Poll,
        timeout: Duration,
    }

    /// A step that only acts, completing immediately.
    fn act(
        label: &'static str,
        enter: impl FnMut(&mut Harness) -> Result<(), String> + 'static,
    ) -> Step {
        Step {
            label,
            enter: Box::new(enter),
            poll: Box::new(|_| Ok(true)),
            timeout: Duration::from_secs(1),
        }
    }

    /// A step that only waits for a condition.
    fn wait(
        label: &'static str,
        secs: f64,
        poll: impl FnMut(&mut Harness) -> Result<bool, String> + 'static,
    ) -> Step {
        Step {
            label,
            enter: Box::new(|_| Ok(())),
            poll: Box::new(poll),
            timeout: Duration::from_secs_f64(secs),
        }
    }

    /// A step that acts, then waits for the act to take effect.
    fn step(
        label: &'static str,
        secs: f64,
        enter: impl FnMut(&mut Harness) -> Result<(), String> + 'static,
        poll: impl FnMut(&mut Harness) -> Result<bool, String> + 'static,
    ) -> Step {
        Step {
            label,
            enter: Box::new(enter),
            poll: Box::new(poll),
            timeout: Duration::from_secs_f64(secs),
        }
    }

    /// A step that simply lets `secs` of wall time pass — used where the
    /// assertion is "nothing changed", which needs a window to not
    /// change in.
    fn settle(label: &'static str, secs: f64) -> Step {
        let until = Rc::new(RefCell::new(None::<Instant>));
        let set = Rc::clone(&until);
        Step {
            label,
            enter: Box::new(move |_| {
                *set.borrow_mut() = Some(Instant::now() + Duration::from_secs_f64(secs));
                Ok(())
            }),
            poll: Box::new(move |_| {
                Ok(until
                    .borrow()
                    .is_some_and(|deadline| Instant::now() >= deadline))
            }),
            timeout: Duration::from_secs_f64(secs + 2.0),
        }
    }

    struct Scenario {
        name: &'static str,
        steps: Vec<Step>,
    }

    // ---- the scenarios ----------------------------------------------

    /// Sum of frames imported across every pane — the runner's "did
    /// anything actually reach the screen" counter.
    fn total_frames(h: &Harness) -> u64 {
        h.panes.iter().map(|p| p.frames).sum()
    }

    /// Baseline every scenario starts from: windowed, `BASE_SIZE`, one
    /// fresh engine attached, nothing loaded.
    fn baseline() -> Vec<Step> {
        vec![
            act("reset", |h| {
                h.set_fullscreen(false);
                let _ = h
                    .window
                    .request_inner_size(PhysicalSize::new(BASE_SIZE.0, BASE_SIZE.1));
                h.set_panes(1)
            }),
            settle("settle", 0.3),
            act("relayout", |h| {
                h.relayout();
                Ok(())
            }),
        ]
    }

    /// Frames must keep arriving *and* converge on the window's current
    /// size after each resize — the check that `set_export_size` reaches
    /// the render thread and that the pool retires wrong-size buffers.
    fn scenario_resize() -> Scenario {
        let mut steps = baseline();
        steps.push(act("load looping clip", |h| h.load(0, 1, true)));
        steps.push(wait("first frame", 10.0, |h| Ok(h.panes[0].frames > 0)));
        // A sweep including a shrink, a grow, and a non-multiple-of-8
        // size — mpv's scalers and the D3D/dmabuf allocators all have
        // alignment opinions, and an odd size flushes them out.
        for size in [(640, 360), (1280, 720), (513, 289), (800, 600), (320, 240)] {
            steps.push(step(
                "resize",
                8.0,
                move |h| {
                    let _ = h
                        .window
                        .request_inner_size(PhysicalSize::new(size.0, size.1));
                    Ok(())
                },
                |h| {
                    if let Some(e) = &h.playback_error {
                        return Err(e.clone());
                    }
                    // The window manager is free to refuse an exact
                    // size, so the invariant is "delivered size tracks
                    // the window", not "delivered size == requested".
                    Ok(h.panes[0].last_size == Some(h.window_size()))
                },
            ));
        }
        Scenario {
            name: "resize",
            steps,
        }
    }

    /// Fullscreen is the resize case the compositor drives instead of
    /// us: the window changes size without a `request_inner_size`, so it
    /// exercises the `Resized`-event path end to end.
    fn scenario_fullscreen() -> Scenario {
        let mut steps = baseline();
        steps.push(act("load looping clip", |h| h.load(0, 1, true)));
        steps.push(wait("first frame", 10.0, |h| Ok(h.panes[0].frames > 0)));
        let windowed = Rc::new(RefCell::new((0u32, 0u32)));
        let remember = Rc::clone(&windowed);
        steps.push(step(
            "enter fullscreen",
            10.0,
            move |h| {
                *remember.borrow_mut() = h.window_size();
                h.set_fullscreen(true);
                Ok(())
            },
            {
                let windowed = Rc::clone(&windowed);
                move |h| {
                    let now = h.window_size();
                    if now == *windowed.borrow() {
                        return Ok(false); // compositor hasn't resized us yet
                    }
                    Ok(h.panes[0].last_size == Some(now))
                }
            },
        ));
        steps.push(wait("frames while fullscreen", 5.0, {
            let mut base = None;
            move |h| {
                let base = *base.get_or_insert(h.panes[0].frames);
                Ok(h.panes[0].frames >= base + 10)
            }
        }));
        steps.push(step(
            "leave fullscreen",
            10.0,
            |h| {
                h.set_fullscreen(false);
                Ok(())
            },
            {
                let windowed = Rc::clone(&windowed);
                move |h| {
                    let now = h.window_size();
                    if now != *windowed.borrow() {
                        return Ok(false);
                    }
                    Ok(h.panes[0].last_size == Some(now))
                }
            },
        ));
        Scenario {
            name: "fullscreen",
            steps,
        }
    }

    /// Transport: playing advances the clock, pause freezes it *without*
    /// killing forced re-renders, stop unloads, and a load after a stop
    /// starts clean.
    fn scenario_transport() -> Scenario {
        let mut steps = baseline();
        steps.push(act("load looping clip", |h| h.load(0, 1, true)));
        steps.push(wait("first frame", 10.0, |h| Ok(h.panes[0].frames > 0)));
        steps.push(wait("clock advances", 10.0, {
            let mut first = None;
            move |h| {
                let now = h.panes[0].engine.position().unwrap_or(0.0);
                let first = *first.get_or_insert(now);
                Ok(now > first + 0.10)
            }
        }));

        steps.push(act("pause", |h| {
            h.panes[0]
                .engine
                .set_paused(true)
                .map_err(|e| format!("pause: {e}"))
        }));
        steps.push(settle("settle paused", 0.4));
        let paused_at = Rc::new(RefCell::new(0.0f64));
        let record = Rc::clone(&paused_at);
        steps.push(act("record position", move |h| {
            *record.borrow_mut() = h.panes[0].engine.position().unwrap_or(0.0);
            Ok(())
        }));
        steps.push(settle("hold", 0.5));
        steps.push(wait("clock frozen while paused", 2.0, {
            let paused_at = Rc::clone(&paused_at);
            move |h| {
                if !h.panes[0].engine.is_paused() {
                    return Err("engine reports not paused".into());
                }
                let now = h.panes[0].engine.position().unwrap_or(0.0);
                let was = *paused_at.borrow();
                if (now - was).abs() > 0.20 {
                    return Err(format!("position moved while paused: {was:.3} -> {now:.3}"));
                }
                Ok(true)
            }
        }));
        // The paused-but-still-renderable property: a size change must
        // still publish, or a resize of a paused player shows nothing.
        steps.push(step(
            "forced render while paused",
            10.0,
            |h| {
                let (w, hgt) = h.window_size();
                h.panes[0]
                    .engine
                    .set_export_size(w - 40, hgt - 40)
                    .map_err(|e| format!("set_export_size: {e}"))
            },
            |h| {
                let (w, hgt) = h.window_size();
                Ok(h.panes[0].last_size == Some((w - 40, hgt - 40)))
            },
        ));
        steps.push(act("restore size", |h| {
            h.relayout();
            Ok(())
        }));

        steps.push(act("unpause", |h| {
            h.panes[0]
                .engine
                .set_paused(false)
                .map_err(|e| format!("unpause: {e}"))
        }));
        steps.push(wait("clock advances again", 10.0, {
            let mut first = None;
            move |h| {
                let now = h.panes[0].engine.position().unwrap_or(0.0);
                let first = *first.get_or_insert(now);
                Ok(now > first + 0.10)
            }
        }));

        steps.push(act("stop", |h| {
            h.panes[0].engine.stop().map_err(|e| format!("stop: {e}"))
        }));
        steps.push(wait("idle after stop", 10.0, |h| {
            Ok(h.panes[0].engine.is_idle())
        }));

        // Start again after a stop: a fresh load on the same engine and
        // the same attached backend.
        steps.push(step(
            "start a different clip",
            15.0,
            |h| h.load(0, 2, true),
            |h| {
                if h.panes[0].engine.is_idle() {
                    return Ok(false);
                }
                let want = h.clips[2].secs;
                match h.panes[0].engine.duration() {
                    Some(d) if (d - want).abs() < 0.5 => Ok(true),
                    _ => Ok(false),
                }
            },
        ));
        steps.push(wait("frames from the restarted clip", 10.0, {
            let mut base = None;
            move |h| {
                let base = *base.get_or_insert(h.panes[0].frames);
                Ok(h.panes[0].frames >= base + 10)
            }
        }));
        Scenario {
            name: "play/pause/stop/start",
            steps,
        }
    }

    /// Resource return, in two parts. First the pool: hold every buffer
    /// and delivery must stop dead (no growth past `pool_size`), then
    /// release them and delivery must resume — the pool handed its
    /// buffers back. Then teardown *without* stopping playback:
    /// `detach_render` with frames outstanding and mpv still rolling
    /// must not hang, and a re-attach must come back live.
    fn scenario_teardown() -> Scenario {
        let mut steps = baseline();
        steps.push(act("load looping clip", |h| h.load(0, 1, true)));
        steps.push(wait("first frame", 10.0, |h| Ok(h.panes[0].frames > 0)));

        steps.push(act("hold the whole pool", |h| {
            // The last imported texture *is* a pool buffer on macOS and
            // Windows — those imports park the buffer until wgpu drops
            // the texture and only then return it — while on Linux the
            // import retires the buffer and the pool replaces it.
            // Release it (and drain wgpu so the return actually lands)
            // or one pool slot stays pinned and the hold below can never
            // reach POOL_SIZE raw frames.
            h.panes[0].last = None;
            let _ = h.gfx.device.poll(wgpu::PollType::wait_indefinitely());
            h.panes[0].hold_target = POOL_SIZE;
            Ok(())
        }));
        steps.push(wait("pool exhausted", 10.0, |h| {
            Ok(h.panes[0].held.len() >= POOL_SIZE)
        }));
        let frames_at_exhaustion = Rc::new(RefCell::new(0u64));
        let record = Rc::clone(&frames_at_exhaustion);
        steps.push(act("record", move |h| {
            *record.borrow_mut() = h.panes[0].frames;
            Ok(())
        }));
        steps.push(settle("hold the pool down", 0.6));
        steps.push(wait("no delivery past the pool", 2.0, {
            let frames_at_exhaustion = Rc::clone(&frames_at_exhaustion);
            move |h| {
                let pane = &h.panes[0];
                if pane.over_acquires > 0 {
                    return Err(format!(
                        "pool grew past {POOL_SIZE}: {} extra buffer(s) delivered while every \
                         buffer was held",
                        pane.over_acquires
                    ));
                }
                if pane.frames != *frames_at_exhaustion.borrow() {
                    return Err("frames imported while the pool was fully held".into());
                }
                Ok(true)
            }
        }));
        steps.push(step(
            "release the pool",
            10.0,
            |h| {
                let pane = &mut h.panes[0];
                pane.hold_target = 0;
                // Dropping the frames is what returns (macOS) or retires
                // (Linux/Windows) the buffers.
                pane.held.clear();
                Ok(())
            },
            {
                let frames_at_exhaustion = Rc::clone(&frames_at_exhaustion);
                move |h| Ok(h.panes[0].frames >= *frames_at_exhaustion.borrow() + 10)
            },
        ));

        // Teardown while the video is still rolling, with an imported
        // texture still alive: `last` holds memory the pool no longer
        // owns, and detach must neither block on it nor free it.
        steps.push(step(
            "detach while playing",
            15.0,
            |h| {
                if h.panes[0].engine.is_paused() {
                    return Err("expected playback to still be running".into());
                }
                h.panes[0].engine.detach_render();
                Ok(())
            },
            |h| {
                if h.panes[0].engine.attached_render().is_some() {
                    return Ok(false);
                }
                if h.panes[0].last.is_none() {
                    return Err("no imported texture survived the detach".into());
                }
                Ok(true)
            },
        ));
        steps.push(settle("keep presenting the orphaned texture", 0.5));
        // Record what mpv did with the *file* — freeing the render
        // context pulls the VO out from under a playing file, and
        // libmpv answers by failing it (`MPV_ERROR_VO_INIT_FAILED`) and
        // going idle. That is mpv's contract, not a leak, so the
        // harness records it rather than asserting either way: a shell
        // that wants playback to survive a detach must pause and reload,
        // and this line is the evidence for that advice.
        steps.push(act("record how mpv took the detach", |h| {
            let idle = h.panes[0].engine.is_idle();
            let why = h.playback_error.take();
            h.note(match (idle, why) {
                (true, Some(e)) => format!("detach mid-playback unloaded the file ({e})"),
                (true, None) => "detach mid-playback unloaded the file".into(),
                (false, _) => "playback survived the detach".into(),
            });
            Ok(())
        }));
        steps.push(step(
            "re-attach and reload",
            15.0,
            |h| {
                let rect = h.panes[0].rect;
                let on_update = {
                    let proxy = h.proxy.clone();
                    move || {
                        let _ = proxy.send_event(());
                    }
                };
                h.panes[0]
                    .engine
                    .attach_exported_render(ExportOptions::new(rect.w, rect.h), on_update)
                    .map_err(|e| format!("re-attach: {e}"))?;
                // The engine must be *reusable*, which is the real
                // resource question: a leaked render context, GL context
                // or pool would surface as a failed attach above or as a
                // load that never produces frames below.
                h.load(0, 1, true)
            },
            {
                let mut base = None;
                move |h| {
                    if let Some(e) = &h.playback_error {
                        return Err(format!("reload after re-attach failed: {e}"));
                    }
                    let base = *base.get_or_insert(h.panes[0].frames);
                    Ok(h.panes[0].frames >= base + 10)
                }
            },
        ));
        Scenario {
            name: "teardown-while-playing",
            steps,
        }
    }

    /// Back-to-back loads on one engine with no `stop` between them:
    /// each `load` must replace the previous file cleanly, which
    /// `duration` proves (the clips have distinct lengths) and the frame
    /// counter proves is live.
    fn scenario_back_to_back() -> Scenario {
        let mut steps = baseline();
        for clip in 0..CLIP_SPECS.len() {
            steps.push(step(
                "load next clip",
                15.0,
                move |h| h.load(0, clip, true),
                move |h| {
                    if let Some(e) = &h.playback_error {
                        return Err(e.clone());
                    }
                    let want = h.clips[clip].secs;
                    match h.panes[0].engine.duration() {
                        Some(d) if (d - want).abs() < 0.5 => Ok(true),
                        _ => Ok(false),
                    }
                },
            ));
            steps.push(wait("frames from it", 10.0, {
                let mut base = None;
                move |h| {
                    let base = *base.get_or_insert(h.panes[0].frames);
                    Ok(h.panes[0].frames >= base + 10)
                }
            }));
        }
        Scenario {
            name: "back-to-back loads",
            steps,
        }
    }

    /// Several engines, several export backends, one window. Each has
    /// its own hidden GL context, render thread and pool; the harness
    /// composites their frames into separate rectangles of one
    /// swapchain image. This is where per-process GPU resource limits
    /// and any cross-engine interference would show up.
    fn scenario_multi() -> Scenario {
        let mut steps = baseline();
        steps.push(act("two panes", |h| h.set_panes(2)));
        steps.push(act("relayout", |h| {
            h.relayout();
            Ok(())
        }));
        steps.push(act("load both", |h| {
            h.load(0, 1, true)?;
            h.load(1, 2, true)
        }));
        steps.push(wait("both deliver", 20.0, |h| {
            if let Some(e) = &h.playback_error {
                return Err(e.clone());
            }
            Ok(h.panes.iter().all(|p| p.frames >= 30))
        }));
        steps.push(wait("both keep their own clip", 5.0, |h| {
            for (i, clip) in [(0usize, 1usize), (1, 2)] {
                let want = h.clips[clip].secs;
                match h.panes[i].engine.duration() {
                    Some(d) if (d - want).abs() < 0.5 => {}
                    other => {
                        return Err(format!(
                            "pane {i} should be playing {} ({want}s), duration reads {other:?}",
                            h.clips[clip].label
                        ));
                    }
                }
            }
            Ok(true)
        }));
        // A resize with two live backends: both pools must retire and
        // reallocate concurrently.
        steps.push(step(
            "resize with both live",
            15.0,
            |h| {
                let _ = h.window.request_inner_size(PhysicalSize::new(1100, 620));
                Ok(())
            },
            |h| {
                let rects = h.layout(h.panes.len());
                Ok(h.panes
                    .iter()
                    .zip(rects)
                    .all(|(p, r)| p.last_size == Some((r.w, r.h))))
            },
        ));
        Scenario {
            name: "multi-engine one window",
            steps,
        }
    }

    /// Churn iterations measured after the warm-up. Override with
    /// `MPV_HARNESS_LEAK_CYCLES` for a soak: a real leak grows linearly
    /// with cycles, while allocator arenas and driver caches flatten
    /// out, so comparing per-cycle growth at 8 and at 40 is what tells
    /// the two apart.
    const LEAK_ITERATIONS_DEFAULT: usize = 16;

    fn leak_iterations() -> usize {
        std::env::var("MPV_HARNESS_LEAK_CYCLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(LEAK_ITERATIONS_DEFAULT)
    }
    /// Iterations run before the baseline sample. The first attach pulls
    /// in the GL/D3D/Vulkan ICD, mpv's codec tables and wgpu's caches;
    /// none of that is a leak, and all of it lands in the first cycle.
    const LEAK_WARMUP: usize = 6;
    /// Budgets for the handle-like counters, as **totals for the whole
    /// run, not per cycle**. Every kernel handle, fd and GDI/USER object
    /// the export tier takes is closed by an explicit `Drop`, so the
    /// correct answer is zero growth no matter how many cycles run — a
    /// per-cycle allowance would scale the budget with the run length
    /// and quietly permit a genuine leak. They are sized from the
    /// observed noise floor instead: these counters are process-wide,
    /// and the window, the swapchain and ffmpeg move them by a few
    /// either way between samples. Growth that is actually per-cycle
    /// clears this easily over a default run, and a longer run makes a
    /// real leak *more* visible rather than buying it more headroom.
    const LEAK_BUDGET_HANDLES_TOTAL: i64 = 8;
    const LEAK_BUDGET_FDS_TOTAL: i64 = 8;
    /// Looser than the others on purpose: this harness owns a real
    /// window and swapchain, and resizing, going fullscreen and
    /// reconfiguring the surface move GDI/USER objects around for
    /// reasons that have nothing to do with the engine. Measured with
    /// `examples/handle_probe.rs -- attach`, which owns no window, the
    /// attach/detach path is flat (+3 over 40 cycles, unchanged from
    /// cycle 20 on) — so treat a trip here as "go measure it with the
    /// probe", not as a confirmed leak.
    const LEAK_BUDGET_GUI_TOTAL: i64 = 16;
    /// Small but non-zero: every thread this crate spawns is joined by
    /// an explicit teardown, so the correct per-run answer is zero —
    /// the headroom is for pool threads the process shares (GCD workers
    /// on macOS, mpv's own worker pool) parking and unparking between
    /// samples.
    const LEAK_BUDGET_THREADS_TOTAL: i64 = 4;
    const LEAK_BUDGET_RSS_MIB: f64 = 2.0;
    /// Per-cycle growth below which the trend is not worth a verdict —
    /// allocator noise lives here.
    const LEAK_NEGLIGIBLE_MIB: f64 = 0.2;
    /// Quiet time between a teardown and the sample that follows it.
    /// Threads exit and deferred GPU/driver frees land asynchronously,
    /// so a sample taken immediately after a teardown reads *low* — and
    /// a low baseline turns ordinary settling into apparent per-cycle
    /// growth. This was worth a full second of every cycle: at 0.3s the
    /// same run reported handles -3 and +10 on consecutive attempts.
    const SETTLE_BEFORE_SAMPLE: f64 = 1.0;

    /// What each churn cycle tears down. Running both and comparing is
    /// what attributes growth: `Attach` exercises only the code this
    /// crate owns (GL context, pool, shared handles, render thread),
    /// while `Engine` adds a full mpv core create/destroy on top. Growth
    /// that shows in `Engine` but not in `Attach` is mpv's, not ours.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Churn {
        /// New `Engine` every cycle.
        Engine,
        /// One `Engine` for the whole run; attach/detach the exported
        /// backend around it.
        Attach,
        /// One `Engine`, attached once, never detached — only the file
        /// is reloaded. The control: anything that grows here is mpv
        /// accumulating on a long-lived core (demuxer caches, codec
        /// state), not this crate's attach/detach path. Subtract it from
        /// `Attach` to get what the backend is actually responsible for.
        Load,
    }

    /// One attach → load → play → resize → hold → detach cycle,
    /// repeated, with OS resource counters sampled either side — a
    /// leaked GL context, D3D device, shared handle, dmabuf fd, hidden
    /// window or render thread accumulates into something measurable.
    fn scenario_leaks(churn: Churn) -> Scenario {
        let iterations = leak_iterations();
        let mut steps = baseline();
        let baseline_sample = Rc::new(RefCell::new(Resources::default()));
        // Sampled with an engine *alive*, and with none at all: their
        // difference is the sampler's sensitivity — see "compare".
        let live_sample = Rc::new(RefCell::new(Resources::default()));
        let idle_sample = Rc::new(RefCell::new(Resources::default()));
        // One sample per measured cycle. A leak grows at a steady rate
        // per cycle; allocator arenas and driver caches grow fast at
        // first and then flatten — so comparing the first half's rate
        // with the second half's separates them inside a single run.
        let trail: Rc<RefCell<Vec<Resources>>> = Rc::new(RefCell::new(Vec::new()));

        // `cycle` is the unit of churn: everything a shell does over one
        // video's life, including a resize and a held frame, so a leak
        // in any of those paths accumulates.
        let cycle = move |steps: &mut Vec<Step>, clip: usize| {
            match churn {
                Churn::Load => {}
                Churn::Engine => steps.push(act("fresh engine", |h| h.set_panes(1))),
                Churn::Attach => steps.push(act("attach", |h| {
                    // A guard, not an index: a harness that panics tells
                    // you far less than one that fails a named step.
                    if h.panes.is_empty() {
                        return Err("no pane to attach to".into());
                    }
                    let rect = h.panes[0].rect;
                    let on_update = {
                        let proxy = h.proxy.clone();
                        move || {
                            let _ = proxy.send_event(());
                        }
                    };
                    h.panes[0]
                        .engine
                        .attach_exported_render(ExportOptions::new(rect.w, rect.h), on_update)
                        .map_err(|e| format!("attach: {e}"))
                })),
            }
            steps.push(act("relayout", |h| {
                h.relayout();
                Ok(())
            }));
            steps.push(step(
                "load and play",
                20.0,
                move |h| {
                    // Per-cycle counters. In `Attach` mode the pane
                    // outlives the cycle, so without this the previous
                    // cycle's frames would satisfy the poll instantly.
                    let pane = &mut h.panes[0];
                    pane.frames = 0;
                    pane.last_size = None;
                    // A detach mid-playback fails the file (see the
                    // teardown scenario); that event can arrive a tick
                    // after the detach step cleared it, so clear it here
                    // too — right before the load that supersedes it. A
                    // load that then genuinely fails shows up as this
                    // step timing out, with `diagnose` naming the error.
                    h.playback_error = None;
                    h.load(0, clip, true)
                },
                |h| Ok(h.panes[0].frames >= 5),
            ));
            // Deliberately no resize here. A resize retires the whole
            // pool and allocates a fresh one, and a *newly created*
            // shared texture that a consumer then opens strands one
            // reference on the producing D3D11 device until that device
            // is destroyed (a driver behaviour, not our bookkeeping —
            // see `examples/handle_probe.rs`). That is bounded by pool
            // churn rather than by frames, but mixing it in here would
            // make all three variants grow for a reason none of them is
            // trying to isolate. The resize path has its own scenario,
            // and `handle_probe -- resize` measures it directly.
            // Hold a frame across the teardown: its buffer leaves the
            // pool and must still be released when the frame drops.
            steps.push(act("hold a frame", |h| {
                h.panes[0].hold_target = 1;
                Ok(())
            }));
            steps.push(wait("frame held", 20.0, |h| {
                Ok(!h.panes[0].held.is_empty())
            }));
            match churn {
                Churn::Engine => steps.push(act("tear the engine down", |h| {
                    // Dropping the pane drops the held frame *and* the
                    // engine: detach joins the render thread, then the
                    // engine's own Drop takes the mpv core down.
                    for pane in h.panes.drain(..) {
                        pane.engine.detach_render();
                    }
                    h.playback_error = None;
                    Ok(())
                })),
                Churn::Attach => steps.push(act("detach, keep the engine", |h| {
                    let pane = &mut h.panes[0];
                    pane.hold_target = 0;
                    // Released before the detach so the buffer goes back
                    // through the pool rather than out with the context.
                    pane.held.clear();
                    pane.last = None;
                    pane.engine.detach_render();
                    h.playback_error = None;
                    Ok(())
                })),
                Churn::Load => steps.push(act("stop, keep engine and attach", |h| {
                    let pane = &mut h.panes[0];
                    pane.hold_target = 0;
                    pane.held.clear();
                    pane.last = None;
                    let r = pane.engine.stop().map_err(|e| format!("stop: {e}"));
                    h.playback_error = None;
                    r
                })),
            }
            // wgpu frees dropped resources on its own schedule, so an
            // unpolled device makes the *harness* look like the leak.
            // Drain it before the cycle's sample lands.
            steps.push(act("drain the wgpu device", |h| {
                let _ = h.gfx.device.poll(wgpu::PollType::wait_indefinitely());
                Ok(())
            }));
            steps.push(settle("let handles wind down", SETTLE_BEFORE_SAMPLE));
        };

        if churn == Churn::Attach {
            steps.push(act("detach the baseline attach", |h| {
                h.panes[0].engine.detach_render();
                Ok(())
            }));
        }
        for i in 0..LEAK_WARMUP {
            cycle(&mut steps, i % CLIP_SPECS.len());
        }
        // Sensitivity probe, in two halves: a sample with *no* engine,
        // then one with an engine up and playing. Their difference is
        // what proves the counters respond to the thing under test —
        // without it the whole scenario is vacuous, because a sampler
        // that reads nothing reports a perfect zero delta and passes
        // forever. Measured this way rather than against the baseline
        // so it means the same thing in all three churn modes (in
        // `Load` the baseline already *has* a live engine).
        steps.push(act("tear everything down", |h| {
            for pane in h.panes.drain(..) {
                pane.engine.detach_render();
            }
            let _ = h.gfx.device.poll(wgpu::PollType::wait_indefinitely());
            h.playback_error = None;
            Ok(())
        }));
        steps.push(settle("let handles wind down", SETTLE_BEFORE_SAMPLE));
        {
            let record = Rc::clone(&idle_sample);
            steps.push(act("idle sample", move |_| {
                *record.borrow_mut() = sample();
                Ok(())
            }));
        }
        steps.push(act("probe engine", |h| h.set_panes(1)));
        steps.push(act("relayout", |h| {
            h.relayout();
            Ok(())
        }));
        steps.push(step(
            "probe playing",
            20.0,
            |h| h.load(0, 0, true),
            |h| Ok(h.panes[0].frames >= 5),
        ));
        {
            let record = Rc::clone(&live_sample);
            steps.push(act("live sample", move |_| {
                *record.borrow_mut() = sample();
                Ok(())
            }));
        }
        // Leave the process in the same shape the baseline sample was
        // taken in, or the comparison is not apples to apples: no engine
        // at all for `Engine` churn, one detached engine for `Attach`.
        steps.push(act("drop the probe", move |h| {
            match churn {
                Churn::Engine => {
                    for pane in h.panes.drain(..) {
                        pane.engine.detach_render();
                    }
                }
                Churn::Attach => {
                    if let Some(pane) = h.panes.first_mut() {
                        pane.hold_target = 0;
                        pane.held.clear();
                        pane.last = None;
                        pane.engine.detach_render();
                    }
                }
                Churn::Load => {
                    if let Some(pane) = h.panes.first_mut() {
                        pane.hold_target = 0;
                        pane.held.clear();
                        pane.last = None;
                        let _ = pane.engine.stop();
                    }
                }
            }
            h.playback_error = None;
            Ok(())
        }));
        steps.push(settle("let the OS settle", 0.3));

        steps.push(settle("let handles wind down", SETTLE_BEFORE_SAMPLE));
        {
            // Taken *after* the probe is dropped, in exactly the state
            // every cycle ends in — otherwise the first cycle's delta
            // includes the difference between two different states.
            let record = Rc::clone(&baseline_sample);
            steps.push(act("baseline sample", move |_| {
                *record.borrow_mut() = sample();
                Ok(())
            }));
        }
        {
            let trail = Rc::clone(&trail);
            steps.push(act("start the trail", move |_| {
                trail.borrow_mut().push(sample());
                Ok(())
            }));
        }
        for i in 0..iterations {
            cycle(&mut steps, i % CLIP_SPECS.len());
            let trail = Rc::clone(&trail);
            steps.push(act("sample the cycle", move |_| {
                trail.borrow_mut().push(sample());
                Ok(())
            }));
        }
        {
            let baseline_sample = Rc::clone(&baseline_sample);
            let live_sample = Rc::clone(&live_sample);
            let idle_sample = Rc::clone(&idle_sample);
            steps.push(act("compare", move |h| {
                let iterations = iterations;
                let before = *baseline_sample.borrow();
                if before.is_empty() {
                    h.note(format!(
                        "no resource sampler on {} — nothing measured",
                        std::env::consts::OS
                    ));
                    return Ok(());
                }
                // The counters must actually respond to an engine
                // existing, or "no growth" means nothing.
                let sensitivity = live_sample.borrow().since(*idle_sample.borrow());
                let responsive = [
                    sensitivity.handles,
                    sensitivity.fds,
                    sensitivity.gui,
                    sensitivity.threads,
                    sensitivity.rss,
                ]
                .iter()
                .any(|d| d.is_some_and(|d| d > 0));
                if !responsive {
                    return Err(format!(
                        "resource sampler is not sensitive: a live engine moved none of the                          counters ({sensitivity}) — the leak check would pass vacuously"
                    ));
                }

                let deltas = sample().since(before);
                let n = iterations as f64;
                // Whether the memory *trend* (below) judged this a leak.
                // `None` until enough samples exist to trend, in which case
                // the RSS check falls back to the flat per-cycle budget.
                let mut rss_is_leak: Option<bool> = None;
                // Linear or flattening? Compare the two halves' rates.
                let trail = trail.borrow();
                if let (Some(first), Some(last)) = (trail.first(), trail.last())
                    && trail.len() >= 4
                {
                    let mid_index = trail.len() / 2;
                    let mid = trail[mid_index];
                    let mib = |d: Deltas| d.rss.unwrap_or(0) as f64 / (1024.0 * 1024.0);
                    let early = mib(mid.since(*first)) / mid_index as f64;
                    let late = mib(last.since(mid)) / (trail.len() - 1 - mid_index) as f64;
                    // Three readings, in order of how much they matter:
                    // too small to care about, shrinking (a cache
                    // settling), or holding steady (what a leak does).
                    let verdict = if late < LEAK_NEGLIGIBLE_MIB {
                        "negligible"
                    } else if late <= early * 0.5 {
                        "flattening (cache/arena, not a leak)"
                    } else if early < LEAK_NEGLIGIBLE_MIB {
                        // Growth only in the second half is a process
                        // that was still settling when the baseline was
                        // taken, not a per-cycle leak — a leak is
                        // present from the first cycle.
                        "late-onset (warm-up, not a leak)"
                    } else {
                        "WARN steady — suspect a leak"
                    };
                    // "WARN steady" is the only verdict that means leak;
                    // the others are warm-up/cache the flat budget must not
                    // punish. This is what the RSS check defers to.
                    rss_is_leak = Some(verdict.starts_with("WARN"));
                    h.note(format!(
                        "mem/cycle {early:.2}MiB early vs {late:.2}MiB late: {verdict}"
                    ));
                }
                h.note(format!(
                    "over {iterations} cycles: {deltas} (a live engine reads {sensitivity})"
                ));

                let mut leaks = Vec::new();
                let mut check = |what: &str, delta: Option<i64>, budget: i64| {
                    if let Some(delta) = delta
                        && delta > budget
                    {
                        let per = delta as f64 / n;
                        leaks.push(format!(
                            "{what} grew by {delta} over {iterations} cycles \
                             ({per:.2}/cycle; budget for the whole run is {budget}, \
                             since these are explicitly closed and must not accumulate)"
                        ));
                    }
                };
                check("handles", deltas.handles, LEAK_BUDGET_HANDLES_TOTAL);
                check("fds", deltas.fds, LEAK_BUDGET_FDS_TOTAL);
                check("gui objects", deltas.gui, LEAK_BUDGET_GUI_TOTAL);
                check("threads", deltas.threads, LEAK_BUDGET_THREADS_TOTAL);
                if let Some(rss) = deltas.rss {
                    let mib = rss as f64 / (1024.0 * 1024.0);
                    // Memory legitimately carries one-time warm-up (ICD,
                    // codec tables, allocator arenas), so the early-vs-late
                    // trend is the authority — a flat total-average budget
                    // over-reports on warm-up-heavy runs. The budget is only
                    // the fallback when there were too few samples to trend.
                    let over = rss_is_leak.unwrap_or(mib > LEAK_BUDGET_RSS_MIB * n);
                    if over {
                        leaks.push(format!(
                            "rss grew by {mib:.1}MiB ({:.2}MiB/cycle) on a non-flattening trend",
                            mib / n
                        ));
                    }
                }
                if leaks.is_empty() {
                    Ok(())
                } else {
                    Err(leaks.join("; "))
                }
            }));
        }
        Scenario {
            name: match churn {
                Churn::Engine => "resource leaks (engine churn)",
                Churn::Attach => "resource leaks (attach churn)",
                Churn::Load => "resource leaks (load churn)",
            },
            steps,
        }
    }

    fn all_scenarios() -> Vec<Scenario> {
        vec![
            scenario_resize(),
            scenario_fullscreen(),
            scenario_transport(),
            scenario_teardown(),
            scenario_back_to_back(),
            scenario_multi(),
            scenario_leaks(Churn::Load),
            scenario_leaks(Churn::Attach),
            scenario_leaks(Churn::Engine),
        ]
    }

    // ---- runner ------------------------------------------------------

    struct Outcome {
        name: &'static str,
        detail: String,
        failure: Option<String>,
    }

    struct Runner {
        scenarios: Vec<Scenario>,
        scenario: usize,
        step: usize,
        entered: bool,
        deadline: Instant,
        started: Instant,
        outcomes: Vec<Outcome>,
    }

    impl Runner {
        fn new(scenarios: Vec<Scenario>) -> Self {
            Self {
                scenarios,
                scenario: 0,
                step: 0,
                entered: false,
                deadline: Instant::now(),
                started: Instant::now(),
                outcomes: Vec::new(),
            }
        }

        /// Abandon the current scenario, recording why.
        fn fail(&mut self, h: &Harness, why: String) {
            let name = self.scenarios[self.scenario].name;
            let step = self.scenarios[self.scenario].steps[self
                .step
                .min(self.scenarios[self.scenario].steps.len().saturating_sub(1))]
            .label;
            self.outcomes.push(Outcome {
                name,
                detail: self.detail(h),
                failure: Some(format!("[{step}] {why}")),
            });
            self.next_scenario();
        }

        fn detail(&self, h: &Harness) -> String {
            // Every scenario rebuilds its panes in `baseline`, which
            // zeroes the counters — so the live total *is* this
            // scenario's tally.
            let mut detail = format!(
                "{} frame(s), {:.1}s",
                total_frames(h),
                self.started.elapsed().as_secs_f64()
            );
            for note in &h.notes {
                detail.push_str("; ");
                detail.push_str(note);
            }
            detail
        }

        fn next_scenario(&mut self) {
            self.scenario += 1;
            self.step = 0;
            self.entered = false;
        }

        /// One tick. Returns `true` once every scenario has finished.
        fn tick(&mut self, h: &mut Harness) -> bool {
            if self.scenario >= self.scenarios.len() {
                return true;
            }
            // Catch any resize the `Resized` event didn't deliver before
            // the step's `poll` reads the window/pane sizes.
            h.reconcile_layout();
            if self.step == 0 && !self.entered {
                self.started = Instant::now();
                h.notes.clear();
            }
            if self.step >= self.scenarios[self.scenario].steps.len() {
                let name = self.scenarios[self.scenario].name;
                self.outcomes.push(Outcome {
                    name,
                    detail: self.detail(h),
                    failure: None,
                });
                self.next_scenario();
                return self.scenario >= self.scenarios.len();
            }

            if !self.entered {
                let step = &mut self.scenarios[self.scenario].steps[self.step];
                self.deadline = Instant::now() + step.timeout;
                self.entered = true;
                if let Err(e) = (step.enter)(h) {
                    self.fail(h, e);
                    return self.scenario >= self.scenarios.len();
                }
            }
            let step = &mut self.scenarios[self.scenario].steps[self.step];
            match (step.poll)(h) {
                Ok(true) => {
                    self.step += 1;
                    self.entered = false;
                }
                Ok(false) => {
                    if Instant::now() >= self.deadline {
                        let secs = step.timeout.as_secs_f64();
                        let state = h.diagnose();
                        self.fail(h, format!("timed out after {secs:.1}s\n      {state}"));
                    }
                }
                Err(e) => self.fail(h, e),
            }
            self.scenario >= self.scenarios.len()
        }

        /// Print the report. Returns the number of failures.
        fn report(&self) -> usize {
            let width = self
                .outcomes
                .iter()
                .map(|o| o.name.len())
                .max()
                .unwrap_or(0)
                .max(24);
            println!();
            for outcome in &self.outcomes {
                let dots = ".".repeat(width + 2 - outcome.name.len());
                match &outcome.failure {
                    None => println!("  {} {} PASS  ({})", outcome.name, dots, outcome.detail),
                    Some(why) => {
                        println!("  {} {} FAIL  ({})", outcome.name, dots, outcome.detail);
                        println!("      {why}");
                    }
                }
            }
            let failed = self.outcomes.iter().filter(|o| o.failure.is_some()).count();
            let passed = self.outcomes.len() - failed;
            println!("\n{passed} passed, {failed} failed\n");
            failed
        }
    }

    // ---- app ---------------------------------------------------------

    enum Mode {
        Scripted(Runner),
        Interactive,
    }

    const KEYS: &str = "[space] pause  [s] stop  [n] next clip  [f] fullscreen\n\
                        [d] detach  [a] re-attach  [1]/[2] panes  [+]/[-] resize  [q] quit";

    struct App {
        harness: Option<Harness>,
        mode: Mode,
        clips: Vec<Clip>,
        proxy: EventLoopProxy<()>,
        next_tick: Instant,
        /// Interactive-only: which clip `n` loads next.
        cursor: usize,
        failures: usize,
        /// The report is printed exactly once: `about_to_wait` keeps
        /// firing after `exit()` until the loop actually unwinds.
        reported: bool,
    }

    impl App {
        /// Print the scenario report if it has not been printed yet.
        fn report_once(&mut self, extra_failures: usize) {
            if self.reported {
                return;
            }
            self.reported = true;
            if let Mode::Scripted(runner) = &self.mode {
                self.failures = runner.report() + extra_failures;
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.harness.is_some() {
                return;
            }
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("mpv-engine harness")
                            .with_inner_size(PhysicalSize::new(BASE_SIZE.0, BASE_SIZE.1)),
                    )
                    .expect("create window"),
            );
            let gfx = gfx_for_window(window.clone());
            let mut harness = Harness {
                window,
                gfx,
                clips: std::mem::take(&mut self.clips),
                panes: Vec::new(),
                proxy: self.proxy.clone(),
                playback_error: None,
                notes: Vec::new(),
                // (0, 0) never matches a real window, so the first tick
                // reconciles the layout even if no `Resized` ever lands.
                layout_size: (0, 0),
            };
            if matches!(self.mode, Mode::Interactive) {
                harness.set_panes(1).expect("attach");
                harness.load(0, 0, true).expect("load");
                println!("{KEYS}");
            }
            self.harness = Some(harness);
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick));
        }

        fn user_event(&mut self, _event_loop: &ActiveEventLoop, (): ()) {
            if let Some(harness) = &mut self.harness {
                harness.pump();
                harness.window.request_redraw();
            }
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            let now = Instant::now();
            if now < self.next_tick {
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick));
                return;
            }
            self.next_tick = now + TICK;
            let Some(harness) = &mut self.harness else {
                return;
            };
            harness.pump();
            // Present every tick, not just on new video: a paused or
            // ended clip still has to survive resizes and occlusion.
            harness.redraw();
            let done = match &mut self.mode {
                Mode::Scripted(runner) => runner.tick(harness),
                Mode::Interactive => false,
            };
            if done {
                self.report_once(0);
                event_loop.exit();
                return;
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick));
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    // A scripted run closed by hand is an abort, not a pass.
                    if matches!(self.mode, Mode::Scripted(_)) && !self.reported {
                        eprintln!("aborted: window closed before the run finished");
                        self.report_once(1);
                    }
                    event_loop.exit();
                }
                WindowEvent::Resized(_) => {
                    if let Some(harness) = &mut self.harness {
                        harness.relayout();
                        harness.window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if let Some(harness) = &mut self.harness {
                        harness.redraw();
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => self.key(event_loop, code),
                _ => {}
            }
        }

        fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
            // Joins every hidden render thread; safe with frames
            // outstanding, and the point of doing it explicitly is that
            // a hang here is a harness failure too, not a silent
            // process-exit race.
            if let Some(harness) = &mut self.harness {
                for pane in harness.panes.drain(..) {
                    pane.engine.detach_render();
                }
            }
        }
    }

    impl App {
        fn key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode) {
            if matches!(code, KeyCode::KeyQ | KeyCode::Escape) {
                event_loop.exit();
                return;
            }
            if !matches!(self.mode, Mode::Interactive) {
                return;
            }
            let Some(h) = &mut self.harness else { return };
            let report = |what: &str, r: Result<(), String>| match r {
                Ok(()) => println!("  {what}"),
                Err(e) => eprintln!("  {what}: {e}"),
            };
            match code {
                KeyCode::Space => {
                    let paused = h.panes[0].engine.is_paused();
                    report(
                        if paused { "play" } else { "pause" },
                        h.panes[0]
                            .engine
                            .set_paused(!paused)
                            .map_err(|e| e.to_string()),
                    );
                }
                KeyCode::KeyS => {
                    report("stop", h.panes[0].engine.stop().map_err(|e| e.to_string()))
                }
                KeyCode::KeyN => {
                    self.cursor = (self.cursor + 1) % CLIP_SPECS.len();
                    let label = h.clips[self.cursor].label;
                    let cursor = self.cursor;
                    report(&format!("load {label}"), h.load(0, cursor, true));
                }
                KeyCode::KeyF => {
                    let on = h.window.fullscreen().is_none();
                    h.set_fullscreen(on);
                    println!("  fullscreen {}", if on { "on" } else { "off" });
                }
                KeyCode::KeyD => {
                    for pane in &h.panes {
                        pane.engine.detach_render();
                    }
                    println!("  detached (the last frame keeps presenting)");
                }
                KeyCode::KeyA => {
                    for pane in &mut h.panes {
                        let on_update = {
                            let proxy = self.proxy.clone();
                            move || {
                                let _ = proxy.send_event(());
                            }
                        };
                        if let Err(e) = pane.engine.attach_exported_render(
                            ExportOptions::new(pane.rect.w, pane.rect.h),
                            on_update,
                        ) {
                            eprintln!("  re-attach: {e}");
                        }
                    }
                    println!("  re-attached");
                }
                KeyCode::Digit1 | KeyCode::Digit2 => {
                    let n = if code == KeyCode::Digit1 { 1 } else { 2 };
                    if let Err(e) = h.set_panes(n) {
                        eprintln!("  panes: {e}");
                        return;
                    }
                    h.relayout();
                    for i in 0..n {
                        let clip = (self.cursor + i) % CLIP_SPECS.len();
                        report(&format!("pane {i}"), h.load(i, clip, true));
                    }
                }
                KeyCode::Equal | KeyCode::NumpadAdd | KeyCode::Minus | KeyCode::NumpadSubtract => {
                    let grow = matches!(code, KeyCode::Equal | KeyCode::NumpadAdd);
                    let (w, hgt) = h.window_size();
                    let scale = |v: u32| {
                        if grow {
                            (v * 5 / 4).min(3840)
                        } else {
                            (v * 4 / 5).max(160)
                        }
                    };
                    let _ = h
                        .window
                        .request_inner_size(PhysicalSize::new(scale(w), scale(hgt)));
                    println!("  resize -> {}x{}", scale(w), scale(hgt));
                }
                _ => {}
            }
        }
    }

    // ---- entry -------------------------------------------------------

    pub fn run() -> i32 {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            println!(
                "usage: harness [--interactive] [scenario-substring...]\n\nscenarios:\n{}\n\n{KEYS}",
                all_scenarios()
                    .iter()
                    .map(|s| format!("  {}", s.name))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            return 0;
        }
        let interactive = args.iter().any(|a| a == "--interactive" || a == "-i");
        let filters: Vec<&str> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(String::as_str)
            .collect();

        let dir = tempfile::tempdir().expect("tempdir");
        let Some(clips) = generate_clips(dir.path()) else {
            // Same contract as the integration suites: missing tooling
            // is a skip, not a failure.
            return 0;
        };

        let mode = if interactive {
            Mode::Interactive
        } else {
            let mut scenarios = all_scenarios();
            if !filters.is_empty() {
                scenarios.retain(|s| filters.iter().any(|f| s.name.contains(f)));
                if scenarios.is_empty() {
                    eprintln!("no scenario matches {filters:?}");
                    return 2;
                }
            }
            println!("running {} scenario(s)", scenarios.len());
            Mode::Scripted(Runner::new(scenarios))
        };

        let event_loop = EventLoop::with_user_event().build().expect("event loop");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            harness: None,
            mode,
            clips,
            proxy,
            next_tick: Instant::now(),
            cursor: 0,
            failures: 0,
            reported: false,
        };
        event_loop.run_app(&mut app).expect("event loop run");
        // `dir` drops here, after every engine is torn down in
        // `exiting` — mpv must not be holding an open file.
        drop(dir);
        i32::from(app.failures != 0)
    }
}

#[cfg(all(
    feature = "wgpu",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn main() {
    std::process::exit(harness::run());
}

#[cfg(not(all(
    feature = "wgpu",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
)))]
fn main() {
    eprintln!("the harness needs the `wgpu` feature on macOS, Linux or Windows");
}
