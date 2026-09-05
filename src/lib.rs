//! # mpv-engine
//!
//! Toolkit-agnostic core for embedding libmpv: handle lifecycle, commands,
//! properties, typed playback events, and render seams. Extracted from two
//! independent embedders — a GTK4 app (GLArea direct-to-FBO) and an iced
//! widget (side-context + readback/GPU export) — after their engine layers
//! converged on the same shape.
//!
//! What belongs here is decided by one rule: **a quirk lives in this crate
//! if its consumer-facing interface survives the upstream fix**; anything
//! whose shape would change with a toolkit's API belongs in that toolkit's
//! adapter crate. See README for the seam map and roadmap.
//!
//! The playback surface covers what a player UI consumes: transport
//! (load — immediate, paused, or deferred until a render context
//! attaches ([`Engine::load_when_ready`]) — pause/seek/stop), the mixer
//! set (volume/mute/speed), typed
//! lifecycle events ([`PlaybackEvent`] — including seek-completion for
//! scrubber snap and end reasons for playlist logic), and push-based
//! property observation ([`Engine::observe`]) for state that polling
//! can't track well (duration becoming known, external pause flips, buffering,
//! video dimensions). Everything else goes through the property/command
//! escape hatches — which speak crate-owned types ([`PropertyValue`],
//! the sealed [`PropertyGet`]), so the underlying binding's traits never
//! enter this crate's public API.
//!
//! The render backends share one attach slot: OpenGL
//! ([`Engine::attach_gl_render`] / [`Engine::render_gl`]) and software
//! ([`Engine::attach_sw_render`] / [`Engine::render_sw`], RGBA into a
//! caller buffer, no GL anywhere). [`Engine::attached_render`] reports
//! which one is live ([`RenderKind`]), and the render-update callback
//! fixed at attach can be replaced afterwards
//! ([`Engine::set_render_update_callback`]) for shells whose real
//! closure only exists once the engine is shared. GL attach takes
//! [`GlRenderOptions`]
//! to fix the shell's render-loop discipline: whether `render_gl` blocks
//! until the frame's target time (right for a toolkit paint handler,
//! wrong on a compositor thread), and mpv's advanced control (which
//! obligates [`Engine::render_update`] after every update callback).
//! On macOS the non-default `export` feature adds a third, fully safe
//! backend (`Engine::attach_exported_render`): the engine renders on a
//! hidden GL context of its own and hands the shell zero-copy
//! IOSurface-backed frames (`ExportedFrame`) for Metal/wgpu import — no
//! GL and no pixel copies in the shell. The `wgpu` feature (implies
//! `export`, wgpu-major-locked) adds `ExportedFrame::into_wgpu_texture`,
//! which wraps the frame as a `wgpu::Texture` on the shell's device and
//! ties the pool buffer's return to wgpu's own GPU-completion tracking.
//! Event delivery is pull-based
//! ([`Engine::pump_events`]) with an optional push signal
//! ([`Engine::set_wakeup_callback`]) for shells that don't want a
//! polling timer.
//!
//! Quirks this crate owns so consumers don't have to rediscover them:
//! - `LC_NUMERIC=C` forced before `mpv_create` (toolkits setlocale behind
//!   your back; mpv's number parsing breaks under comma-decimal locales).
//! - Render context freed strictly before the mpv handle — structural
//!   since rsmpv 0.2 (the context co-owns the core) — and, for OpenGL,
//!   only with the GL context current: an obligation
//!   [`Engine::attach_gl_render`] carries as its `unsafe` contract (see
//!   [`Engine::detach_render`]).
//! - Commands pass pre-tokenized argument arrays (`mpv_command`), so paths
//!   with spaces/quotes need no escaping — pinned by a regression test.
//! - Errored end-of-file surfaces as a typed
//!   [`PlaybackEvent::Failed`] carrying the raw `mpv_error` code (for
//!   integrators mapping to their own error copy) and a diagnostic
//!   message (mpv's `mpv_error_string` text plus the numeric code).
//! - The update callback's threading contract (mpv render thread, plus one
//!   synchronous call at registration) is documented at the seam.
//!
//! No toolkit dependencies, no git dependencies, no main-loop opinions:
//! events are *pumped*, and the update callback is `Send + Sync` —
//! bridging to a main loop is the shell's job, because that part is
//! toolkit-shaped.

#![warn(missing_docs)]

mod engine;
mod error;
#[cfg(all(feature = "export", target_os = "macos"))]
mod export;
mod render;

pub use engine::{
    EndReason, Engine, EngineBuilder, ObserveId, PlaybackEvent, PropertyFormat, PropertyGet,
    PropertyValue,
};
pub use error::{Error, Result};
#[cfg(all(feature = "export", target_os = "macos"))]
pub use export::{ExportOptions, ExportedFrame};
pub use render::{GlRenderOptions, ProcAddressFn, RenderKind};
