//! Minimal winit + wgpu video player over the exported-frame backend:
//!
//! ```sh
//! cargo run --features wgpu --example winit_player -- path/to/video.mkv
//! ```
//!
//! The whole render path is: mpv publishes a frame on the engine's
//! hidden render thread → the update callback pokes the winit event
//! loop → `acquire_frame` + `into_wgpu_texture` → one
//! `copy_texture_to_texture` into the surface. No shaders, no
//! pipelines, no GL anywhere in this file — mpv letterboxes into the
//! frame itself, so the copy is the entire compositor.

#[cfg(wgpu_backend)]
mod player {
    use std::sync::Arc;

    use mpv_engine::{Engine, ExportOptions, PlaybackEvent};
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
    use winit::window::{Window, WindowId};

    struct Gfx {
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,
        // The most recently imported frame, retained so a bare repaint
        // (resize, occlusion, a wakeup with no new video) can re-present
        // it — a paused or ended clip never publishes a replacement.
        last: Option<wgpu::Texture>,
    }

    struct App {
        engine: Engine,
        proxy: EventLoopProxy<()>,
        path: String,
        gfx: Option<Gfx>,
    }

    fn gfx_for_window(window: Arc<Window>) -> Gfx {
        let mut descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()));
        // The backend the frame import lands on, straight from the crate —
        // no per-OS ladder here.
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
        let device_descriptor = wgpu::DeviceDescriptor {
            // Whatever this platform's frame import needs (dmabuf opt-in on
            // Linux, nothing elsewhere) — again from the crate, not cfg'd here.
            required_features: mpv_engine::REQUIRED_WGPU_FEATURES,
            ..Default::default()
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&device_descriptor)).expect("wgpu device");

        let caps = surface.get_capabilities(&adapter);
        assert!(
            caps.formats.contains(&wgpu::TextureFormat::Bgra8Unorm),
            "surface does not offer Bgra8Unorm (got {:?})",
            caps.formats
        );
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            // COPY_DST so the imported frame can be blitted straight in.
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
            window,
            surface,
            device,
            queue,
            config,
            last: None,
        }
    }

    impl App {
        fn resize(&mut self, size: PhysicalSize<u32>) {
            let Some(gfx) = &mut self.gfx else { return };
            gfx.config.width = size.width.max(1);
            gfx.config.height = size.height.max(1);
            gfx.surface.configure(&gfx.device, &gfx.config);
            // Zero pauses rendering (minimized); a real size re-renders
            // promptly even when paused or ended.
            self.engine
                .set_export_size(size.width, size.height)
                .expect("set_export_size");
        }

        fn redraw(&mut self) {
            let engine = &self.engine;
            let Some(gfx) = self.gfx.as_mut() else { return };
            // Consume the newest published frame if one is waiting and
            // keep it as `last`. Do this *before* touching the surface so
            // a wakeup that carries no new video (mpv pokes us for many
            // reasons) doesn't acquire a swapchain image at all.
            match engine.acquire_frame() {
                Ok(Some(frame)) => match frame.into_wgpu_texture(&gfx.device) {
                    Ok(texture) => gfx.last = Some(texture),
                    Err(e) => eprintln!("frame import failed: {e}"),
                },
                Ok(None) => {} // nothing new — we'll re-present `last`
                Err(_) => return,
            }
            // Nothing to show until the first frame has landed.
            let Some(texture) = gfx.last.as_ref() else {
                return;
            };
            // Acquire a swapchain image *only* now that we have something
            // to draw: an image acquired and then dropped without
            // `present` is never returned to the presentation engine, and
            // on Vulkan a few of those starve the swapchain — every later
            // `get_current_texture` times out and the window goes black.
            use wgpu::CurrentSurfaceTexture as Cst;
            let target = match gfx.surface.get_current_texture() {
                Cst::Success(target) | Cst::Suboptimal(target) => target,
                Cst::Timeout | Cst::Occluded => return,
                Cst::Outdated | Cst::Lost | Cst::Validation => {
                    // Reconfigure and retry — `last` is retained, so the
                    // redraw completes even when mpv will never publish
                    // another frame.
                    gfx.surface.configure(&gfx.device, &gfx.config);
                    gfx.window.request_redraw();
                    return;
                }
            };
            // Frame and surface sizes track each other via `resize`; copy
            // the intersection so a stale-size frame mid-resize is shown
            // cropped rather than dropped.
            let extent = wgpu::Extent3d {
                width: texture.width().min(target.texture.width()),
                height: texture.height().min(target.texture.height()),
                depth_or_array_layers: 1,
            };
            let mut encoder = gfx.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_texture(
                texture.as_image_copy(),
                target.texture.as_image_copy(),
                extent,
            );
            gfx.queue.submit([encoder.finish()]);
            gfx.window.pre_present_notify();
            gfx.queue.present(target);
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.gfx.is_some() {
                return;
            }
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes().with_title("mpv-engine"))
                    .expect("create window"),
            );
            let gfx = gfx_for_window(window);

            let size = gfx.window.inner_size();
            let on_update = {
                let proxy = self.proxy.clone();
                move || {
                    let _ = proxy.send_event(());
                }
            };
            self.engine
                .attach_exported_render(
                    ExportOptions::new(size.width.max(1), size.height.max(1)),
                    on_update,
                )
                .expect("attach exported render");
            // Same poke for queued playback events, so `Ended` reaches us
            // even once video updates stop.
            let proxy = self.proxy.clone();
            self.engine.set_wakeup_callback(move || {
                let _ = proxy.send_event(());
            });
            self.engine.load_when_ready(&self.path).expect("load");
            self.gfx = Some(gfx);
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, (): ()) {
            for event in self.engine.pump_events() {
                match event {
                    PlaybackEvent::Ended { .. } | PlaybackEvent::Shutdown => event_loop.exit(),
                    PlaybackEvent::Failed { message, .. } => {
                        eprintln!("playback failed: {message}");
                        event_loop.exit();
                    }
                    _ => {}
                }
            }
            if let Some(gfx) = &self.gfx {
                gfx.window.request_redraw();
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => self.resize(size),
                WindowEvent::RedrawRequested => self.redraw(),
                _ => {}
            }
        }
    }

    pub fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: winit_player <video file>");
            std::process::exit(2);
        };
        // The `video()` preset holds the last frame at EOF (`keep-open`,
        // what a player widget wants); this example instead quits when
        // the clip ends, so put mpv's default back.
        let engine = Engine::video()
            .property("keep-open", "no")
            .build()
            .expect("mpv engine");
        let event_loop = EventLoop::with_user_event().build().expect("event loop");
        let proxy = event_loop.create_proxy();
        let mut app = App {
            engine,
            proxy,
            path,
            gfx: None,
        };
        event_loop.run_app(&mut app).expect("event loop run");
        // Joins the hidden render thread; safe with frames outstanding.
        app.engine.detach_render();
    }
}

#[cfg(wgpu_backend)]
fn main() {
    player::run();
}

#[cfg(not(wgpu_backend))]
fn main() {
    eprintln!("the exported-frame backend needs macOS, Linux or Windows");
}
