use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};

use parking_lot::Mutex;
use rsmpv::{EndFileReason, Event, Format, Mpv, PropertyData, sys};

use crate::error::{Error, Result, describe_code};
use crate::render::{GlRender, GlRenderOptions, ProcAddressFn, RenderBackend, SwRender};

/// Force `LC_NUMERIC=C` exactly once before the first `mpv_create`. mpv
/// refuses to work under a comma-decimal locale (its option/number parsing
/// breaks), and toolkits like GTK call `setlocale()` from the environment
/// during init — so this must run even when the host app never touches
/// locale itself. Both parent projects carried this guard independently;
/// it lives here now.
///
/// The category constant comes from `libc` because its value is
/// platform-specific: `LC_NUMERIC` is 1 on glibc but 4 on BSD/macOS and
/// Windows — a hardcoded 1 would silently set `LC_COLLATE` there.
fn ensure_c_numeric_locale() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        libc::setlocale(libc::LC_NUMERIC, c"C".as_ptr());
    });
}

/// Playback lifecycle notifications drained by [`Engine::pump_events`].
///
/// Non-exhaustive: match with a wildcard arm — new variants are additive,
/// not breaking.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PlaybackEvent {
    /// A file finished loading and playback is starting.
    Loaded,
    /// Playback ended without error. Errored ends arrive as
    /// [`Failed`](Self::Failed) instead.
    Ended {
        /// Why it ended — playlist logic usually advances on
        /// [`EndReason::Eof`] and holds on [`EndReason::Stop`].
        reason: EndReason,
    },
    /// Playback (re)started after a seek completed or a file began
    /// playing: the displayed frame matches `time-pos` again. Snap
    /// scrubber/position UI here, not while a seek is still in flight.
    PlaybackRestart,
    /// An observed property changed ([`Engine::observe`]); also fires
    /// once with the current value right after observation starts.
    PropertyChanged {
        /// Which observation this notification belongs to — matches the
        /// [`ObserveId`] returned by [`Engine::observe`], so two
        /// observations of the same property stay distinguishable.
        id: ObserveId,
        /// The mpv property name, as passed to [`Engine::observe`].
        name: String,
        /// The new value, in the [`PropertyFormat`] the observation chose.
        value: PropertyValue,
    },
    /// The mpv core is shutting down — e.g. a `quit` issued through
    /// [`Engine::command`], or an input binding if input was enabled.
    /// No further events follow; drop the [`Engine`] soon. (A file that
    /// was playing also gets an [`Ended`](Self::Ended) with
    /// [`EndReason::Quit`] first.)
    Shutdown,
    /// mpv aborted playback — unreadable, corrupt, empty, or an
    /// unrecognized/unsupported format.
    Failed {
        /// Raw `client.h` `mpv_error` code (always negative). Match on
        /// this to map failures to your own user-facing error copy.
        code: i32,
        /// Diagnostic text (mpv's `mpv_error_string` plus the code), not
        /// user-facing copy.
        message: String,
    },
}

/// Why playback ended (mpv's `mpv_end_file_reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// Natural end of file.
    Eof,
    /// `stop` command or equivalent — user intent, don't auto-advance.
    Stop,
    /// The player is quitting.
    Quit,
    /// The entry redirected to another (e.g. a playlist file).
    Redirect,
    /// A reason this crate doesn't recognize; carries mpv's raw code.
    Other(i32),
}

fn end_reason(reason: EndFileReason) -> EndReason {
    match reason {
        EndFileReason::Eof => EndReason::Eof,
        EndFileReason::Stop => EndReason::Stop,
        EndFileReason::Quit => EndReason::Quit,
        EndFileReason::Redirect => EndReason::Redirect,
        EndFileReason::Unknown(code) => EndReason::Other(code),
        // `Error` is routed to `Failed` before this runs, and a future
        // named reason (the enum is non_exhaustive) carries no raw code
        // to forward — mpv's reason codes are non-negative, so -1 is
        // recognizably out-of-band until this crate names the variant.
        _ => EndReason::Other(-1),
    }
}

/// Owned property value delivered by
/// [`PropertyChanged`](PlaybackEvent::PropertyChanged).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PropertyValue {
    /// An mpv flag (`MPV_FORMAT_FLAG`).
    Flag(bool),
    /// A 64-bit integer (`MPV_FORMAT_INT64`).
    Int(i64),
    /// A double (`MPV_FORMAT_DOUBLE`).
    Double(f64),
    /// A string (`MPV_FORMAT_STRING`).
    Str(String),
}

/// Map an event's decoded property payload onto this crate's owned value.
/// `None` when the property is unavailable or the format is one this crate
/// doesn't deliver (e.g. `Node`) — those notifications are skipped, not
/// surfaced as bogus values.
fn property_value(data: PropertyData) -> Option<PropertyValue> {
    match data {
        PropertyData::Flag(v) => Some(PropertyValue::Flag(v)),
        PropertyData::Int64(v) => Some(PropertyValue::Int(v)),
        PropertyData::Double(v) => Some(PropertyValue::Double(v)),
        PropertyData::String(s) | PropertyData::OsdString(s) => Some(PropertyValue::Str(s)),
        _ => None,
    }
}

impl From<bool> for PropertyValue {
    fn from(v: bool) -> Self {
        Self::Flag(v)
    }
}
impl From<i64> for PropertyValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for PropertyValue {
    fn from(v: i32) -> Self {
        Self::Int(v.into())
    }
}
impl From<u32> for PropertyValue {
    fn from(v: u32) -> Self {
        Self::Int(v.into())
    }
}
impl From<f64> for PropertyValue {
    fn from(v: f64) -> Self {
        Self::Double(v)
    }
}
impl From<&str> for PropertyValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}
impl From<String> for PropertyValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

mod sealed {
    use crate::error::Result;

    pub trait PropertyGetImpl: Sized {
        fn get_property(mpv: &rsmpv::Mpv, name: &str) -> Result<Self>;
    }

    macro_rules! impl_property_get {
        ($($t:ty),*) => {$(
            impl PropertyGetImpl for $t {
                fn get_property(mpv: &rsmpv::Mpv, name: &str) -> Result<Self> {
                    Ok(mpv.get_property(name)?)
                }
            }
        )*};
    }
    impl_property_get!(bool, i64, f64, String);
}

/// Types a property can be read as: `bool`, `i64`, `f64`, `String`.
/// Sealed to the formats mpv's property API speaks — the binding's own
/// conversion traits are deliberately not part of this crate's API, so
/// a binding major bump can't be a breaking change here.
pub trait PropertyGet: sealed::PropertyGetImpl {}
impl PropertyGet for bool {}
impl PropertyGet for i64 {}
impl PropertyGet for f64 {}
impl PropertyGet for String {}

/// Wire format for [`Engine::observe`] — picks which [`PropertyValue`]
/// variant change notifications carry (mpv coerces where it can).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyFormat {
    /// Deliver as [`PropertyValue::Flag`].
    Flag,
    /// Deliver as [`PropertyValue::Int`].
    Int,
    /// Deliver as [`PropertyValue::Double`].
    Double,
    /// Deliver as [`PropertyValue::Str`].
    Str,
}

/// Handle returned by [`Engine::observe`]: tags that observation's
/// [`PropertyChanged`](PlaybackEvent::PropertyChanged) events and cancels
/// it via [`Engine::unobserve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObserveId(u64);

/// Configures and creates an [`Engine`]. Properties set here are applied
/// before `mpv_initialize`, which some options require.
pub struct EngineBuilder {
    props: Vec<(String, String)>,
}

impl EngineBuilder {
    /// Set an mpv property/option before initialization.
    pub fn property(mut self, name: &str, value: &str) -> Self {
        self.props.push((name.into(), value.into()));
        self
    }

    /// Create the engine: forces `LC_NUMERIC=C`, applies the queued
    /// properties, and runs `mpv_initialize`.
    pub fn build(self) -> Result<Engine> {
        ensure_c_numeric_locale();
        let mut builder = Mpv::builder()?;
        for (name, value) in &self.props {
            builder = builder.set_property(name, value.as_str())?;
        }
        let mpv = Arc::new(builder.build()?);
        Ok(Engine {
            render: Mutex::new(None),
            attach: Mutex::new(()),
            pump: Mutex::new(()),
            next_observe_id: AtomicU64::new(1),
            mpv,
        })
    }
}

/// One embedded mpv player core.
///
/// Toolkit-agnostic by construction: no main-loop integration, no widget,
/// no GL context of its own. A shell attaches a render target with
/// [`attach_gl_render`](Engine::attach_gl_render), forwards the update
/// callback to its own main loop, and drains
/// [`pump_events`](Engine::pump_events) on whatever cadence suits it.
pub struct Engine {
    /// Live render context (either backend), if any. Freed-before-
    /// terminate ordering is structural: the context co-owns the core
    /// through its own `Arc<Mpv>`, so the core cannot terminate under a
    /// live context regardless of field order.
    render: Mutex<Option<RenderBackend>>,
    /// Serializes attach calls: concurrent `mpv_render_context_create`
    /// on one handle violates render.h's threading rules, and slot
    /// re-checks alone can't prevent the double-create. Deliberately not
    /// the `render` lock — the synchronous `on_update` during create
    /// must stay free to touch render methods.
    attach: Mutex<()>,
    /// Keeps each [`pump_events`](Engine::pump_events) drain atomic.
    /// rsmpv's `poll_event` is safe to call concurrently, but concurrent
    /// pollers *split* the stream (each event goes to exactly one
    /// caller) — two racing pumps would tear ordered sequences like
    /// `Loaded` → `Ended` across their result batches.
    pump: Mutex<()>,
    /// Userdata ids handed to `mpv_observe_property`; each observation
    /// gets a fresh one so [`unobserve`](Engine::unobserve) is precise.
    next_observe_id: AtomicU64,
    /// Shared with any live render context, which holds its own clone.
    mpv: Arc<Mpv>,
}

// `Engine: Send + Sync` is auto-derived: rsmpv marks `Mpv` Send + Sync
// (libmpv is thread-safe per client.h; the one caveat, single-waiter
// `mpv_wait_event`, is upheld inside rsmpv's `poll_event`), the render
// contexts are `Send` behind a `Sync` mutex, and the remaining fields
// are sync primitives. Do not add manual `unsafe impl`s — they'd
// silence the compiler if a future field is genuinely `!Send`.

impl Engine {
    /// Neutral builder with no properties preset — for consumers whose
    /// configuration doesn't start from [`video`](Self::video) or
    /// [`headless`](Self::headless). (The presets can also be overridden:
    /// properties apply in call order.)
    pub fn builder() -> EngineBuilder {
        EngineBuilder { props: vec![] }
    }

    /// Builder preset for video playback via the libmpv render API.
    /// Frames appear only after [`attach_gl_render`](Self::attach_gl_render);
    /// see that method's note on load ordering.
    pub fn video() -> EngineBuilder {
        Self::builder()
            .property("vo", "libmpv")
            .property("hwdec", "auto-safe")
            .property("keep-open", "always")
    }

    /// Builder preset for audio-only / headless use (`vo=null`): playback
    /// starts as soon as [`load`](Self::load) runs, no render target needed.
    /// Also what the test suite uses — no display required.
    pub fn headless() -> EngineBuilder {
        Self::builder()
            .property("vo", "null")
            .property("keep-open", "always")
    }

    /// Load a file path or URL and start playback.
    ///
    /// For video engines, prefer [`load_paused`](Self::load_paused) until
    /// the shell's surface is mapped: `loadfile` before a render context
    /// exists leaves mpv with nowhere to send frames (audio plays, video
    /// stays black), and demuxing before the window shows wastes work.
    pub fn load(&self, source: &str) -> Result<()> {
        // rsmpv passes args as an array (`mpv_command`), so paths with
        // spaces/quotes need no escaping here. Do not "simplify" this
        // into a formatted command string — that reintroduces the quoting
        // bug this crate's regression test pins.
        self.mpv.command(&["loadfile", source])?;
        Ok(())
    }

    /// [`load`](Self::load), but paused: pause is set *before* `loadfile`
    /// so demuxing/audio don't start before the shell is ready. Call
    /// [`set_paused`](Self::set_paused)`(false)` on your window-ready
    /// signal. (Setting `pause` at init time instead has a tendency to
    /// hang — this runtime-property ordering is the reliable variant.)
    pub fn load_paused(&self, source: &str) -> Result<()> {
        self.set_paused(true)?;
        self.load(source)
    }

    /// Escape hatch: any mpv command, args passed as an array (no quoting
    /// needed).
    pub fn command(&self, name: &str, args: &[&str]) -> Result<()> {
        let mut argv = Vec::with_capacity(args.len() + 1);
        argv.push(name);
        argv.extend_from_slice(args);
        self.mpv.command(&argv)?;
        Ok(())
    }

    /// Set an mpv property. Accepts `bool` / `i64` / `f64` / `&str` /
    /// `String` (anything [`Into<PropertyValue>`]).
    pub fn set_property(&self, name: &str, value: impl Into<PropertyValue>) -> Result<()> {
        match value.into() {
            PropertyValue::Flag(v) => self.mpv.set_property(name, v)?,
            PropertyValue::Int(v) => self.mpv.set_property(name, v)?,
            PropertyValue::Double(v) => self.mpv.set_property(name, v)?,
            PropertyValue::Str(v) => self.mpv.set_property(name, v)?,
        }
        Ok(())
    }

    /// Read an mpv property as `bool`, `i64`, `f64`, or `String`
    /// ([`PropertyGet`] is sealed to those).
    pub fn get_property<T: PropertyGet>(&self, name: &str) -> Result<T> {
        T::get_property(&self.mpv, name)
    }

    /// Pause (`true`) or resume (`false`) playback.
    pub fn set_paused(&self, paused: bool) -> Result<()> {
        self.set_property("pause", paused)
    }

    /// Best-effort pause-state query; `false` when nothing is loaded yet
    /// — use [`is_idle`](Self::is_idle) to tell "playing" apart from
    /// "nothing to play".
    pub fn is_paused(&self) -> bool {
        self.get_property("pause").unwrap_or(false)
    }

    /// True when no file is loaded (`idle-active`) — distinguishes
    /// "nothing to pause" from [`is_paused`](Self::is_paused) being
    /// `false`.
    pub fn is_idle(&self) -> bool {
        self.get_property("idle-active").unwrap_or(true)
    }

    /// Stop playback and unload the current file. Surfaces as
    /// [`PlaybackEvent::Ended`] with [`EndReason::Stop`].
    pub fn stop(&self) -> Result<()> {
        self.command("stop", &[])
    }

    /// Volume in percent: 0–100 is normal range, above 100 amplifies (up
    /// to mpv's `volume-max`).
    pub fn set_volume(&self, percent: f64) -> Result<()> {
        self.set_property("volume", percent)
    }

    /// Best-effort volume query in percent.
    pub fn volume(&self) -> Option<f64> {
        self.get_property("volume").ok()
    }

    /// Mute (`true`) or unmute (`false`) audio.
    pub fn set_muted(&self, muted: bool) -> Result<()> {
        self.set_property("mute", muted)
    }

    /// Best-effort mute-state query; `false` when nothing is loaded yet.
    pub fn is_muted(&self) -> bool {
        self.get_property("mute").unwrap_or(false)
    }

    /// Playback speed multiplier (`1.0` = normal).
    pub fn set_speed(&self, speed: f64) -> Result<()> {
        self.set_property("speed", speed)
    }

    /// Best-effort speed query.
    pub fn speed(&self) -> Option<f64> {
        self.get_property("speed").ok()
    }

    /// Current playback position in seconds, if a file is loaded.
    pub fn position(&self) -> Option<f64> {
        self.get_property("time-pos").ok()
    }

    /// Total duration in seconds. `None` while mpv is still parsing or for
    /// unknown-duration streams.
    pub fn duration(&self) -> Option<f64> {
        self.get_property("duration").ok()
    }

    /// Seek to an absolute position in seconds.
    pub fn seek_absolute(&self, secs: f64) -> Result<()> {
        self.command("seek", &[&format!("{secs:.3}"), "absolute"])
    }

    /// Seek by a delta in seconds (negative seeks backward).
    pub fn seek_relative(&self, secs: f64) -> Result<()> {
        self.command("seek", &[&format!("{secs:.3}"), "relative"])
    }

    /// Observe a property for changes: matching
    /// [`PlaybackEvent::PropertyChanged`] events arrive via
    /// [`pump_events`](Self::pump_events), starting with one carrying the
    /// current value (handy for initializing UI state). `format` picks
    /// the delivered [`PropertyValue`] variant; mpv coerces where it can.
    ///
    /// Typical player set: `pause` (Flag), `time-pos`/`duration`
    /// (Double), `paused-for-cache` (Flag), `dwidth`/`dheight` (Int).
    pub fn observe(&self, name: &str, format: PropertyFormat) -> Result<ObserveId> {
        let fmt = match format {
            PropertyFormat::Flag => Format::Flag,
            PropertyFormat::Int => Format::Int64,
            PropertyFormat::Double => Format::Double,
            PropertyFormat::Str => Format::String,
        };
        let id = self.next_observe_id.fetch_add(1, Ordering::Relaxed);
        self.mpv.observe_property(id, name, fmt)?;
        Ok(ObserveId(id))
    }

    /// Cancel one observation made with [`observe`](Self::observe).
    pub fn unobserve(&self, id: ObserveId) -> Result<()> {
        self.mpv.unobserve_property(id.0)?;
        Ok(())
    }

    /// Register a callback fired whenever mpv queues new events — the
    /// push alternative to polling [`pump_events`](Self::pump_events) on
    /// a timer, and the only timely signal when no frames are flowing
    /// (audio-only playback, a load failure while paused).
    ///
    /// The callback also fires **once synchronously during this call**
    /// (so the construct → share → register ordering documented on the
    /// attach methods applies here too), and mpv may additionally fire
    /// it spuriously. Treat a wakeup as "check the queue", never "an
    /// event arrived".
    ///
    /// Fires on arbitrary mpv-internal threads — possibly several at
    /// once (hence `Sync`), possibly re-entrantly with other engine
    /// calls: do no work and call no engine methods inside — signal your
    /// main loop and pump from there (the same bridging pattern as the
    /// render-update callback). Replaces any previously registered
    /// wakeup callback.
    pub fn set_wakeup_callback(&self, on_wakeup: impl Fn() + Send + Sync + 'static) {
        // rsmpv owns the closure lifecycle: a replaced callback is freed
        // once its last in-flight invocation finishes (possibly on an
        // mpv-internal thread), and everything is unhooked safely during
        // handle teardown. mpv only invokes the latest registration.
        self.mpv.set_wakeup_callback(on_wakeup);
    }

    /// Drain pending mpv events into typed [`PlaybackEvent`]s. Call on a
    /// timer or after the update callback; never blocks.
    ///
    /// Built on rsmpv's non-blocking `poll_event` (`&self`; internally
    /// serialized against libmpv's one-waiter-per-handle rule). The
    /// engine adds the `pump` lock on top so each drain is atomic —
    /// concurrent pollers would otherwise split the stream, tearing
    /// ordered sequences across callers' batches.
    pub fn pump_events(&self) -> Vec<PlaybackEvent> {
        let _guard = self.pump.lock();
        let mut out = Vec::new();
        while let Some(ev) = self.mpv.poll_event() {
            match ev {
                Event::Shutdown => out.push(PlaybackEvent::Shutdown),
                Event::FileLoaded => out.push(PlaybackEvent::Loaded),
                // An errored end-of-file (bad format, load failure,
                // missing/corrupt/empty data) surfaces as a typed
                // `Failed` carrying mpv's error code — never as a quiet
                // `Ended`.
                Event::EndFile {
                    reason: EndFileReason::Error,
                    error,
                    ..
                } => {
                    // Event errors always map onto a raw libmpv code
                    // (rsmpv decodes them from one); `error` itself is
                    // only absent if mpv violates its own contract of
                    // setting a code on errored ends. Generic backstops
                    // both impossibilities.
                    let code = error
                        .and_then(|e| e.raw_code())
                        .unwrap_or(sys::MPV_ERROR_GENERIC);
                    out.push(PlaybackEvent::Failed {
                        code,
                        message: describe_code(code),
                    });
                }
                Event::EndFile { reason, .. } => out.push(PlaybackEvent::Ended {
                    reason: end_reason(reason),
                }),
                Event::PlaybackRestart => out.push(PlaybackEvent::PlaybackRestart),
                Event::PropertyChange {
                    userdata,
                    name,
                    data,
                } => {
                    // A `None` here means the property became unavailable
                    // (or arrived in a format this crate doesn't deliver)
                    // — skipped rather than delivered as a bogus value,
                    // and the drain keeps going.
                    if let Some(value) = property_value(data) {
                        out.push(PlaybackEvent::PropertyChanged {
                            id: ObserveId(userdata),
                            name,
                            value,
                        });
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Create the OpenGL render context. Call with the target GL context
    /// current (e.g. GTK: in the GLArea `realize` handler).
    ///
    /// `get_proc_address` resolves GL symbols (on Linux, EGL 1.5's
    /// `eglGetProcAddress` covers everything mpv asks for — beware
    /// libepoxy on glvnd builds, which doesn't export core GL symbols as
    /// plain `dlsym`-able functions and makes mpv report
    /// `MPV_ERROR_UNSUPPORTED`). `on_update` fires on **mpv's render
    /// thread** — and once synchronously during this call — to signal "a
    /// new frame wants drawing"; forward it to your main loop and call
    /// [`render_gl`](Self::render_gl) from your draw handler.
    ///
    /// The synchronous first call arrives *before* the context is stored:
    /// a [`render_gl`](Self::render_gl) from inside it no-ops (harmlessly
    /// — mpv re-signals). It runs on the caller's thread but outside the
    /// engine's render lock, so calling back into render methods cannot
    /// deadlock.
    ///
    /// `on_update` typically captures a `Weak` handle to your player
    /// state — construct the `Engine`, wrap it in your `Arc`/shared
    /// structure, *then* attach with the weak-capturing closure, then
    /// load.
    ///
    /// `options` fixes the shell's render-loop discipline at attach:
    /// frame pacing ([`GlRenderOptions::block_for_target_time`]) and
    /// mpv's advanced control ([`GlRenderOptions::advanced_control`],
    /// which obligates [`render_update`](Self::render_update) after every
    /// update callback). [`GlRenderOptions::default`] is mpv's stock
    /// behavior.
    ///
    /// # Safety
    /// GL-context currency is a dynamic, per-call rule the type system
    /// cannot capture (rsmpv's OpenGL constructor is `unsafe` for the
    /// same reason, and this crate forwards the obligation rather than
    /// hiding it): the target GL context must be current on the calling
    /// thread now, on every later [`render_gl`](Self::render_gl) or
    /// [`render_update`](Self::render_update), and when the context is
    /// freed — [`detach_render`](Self::detach_render) or the engine's
    /// drop. Violating the rule is undefined behavior.
    pub unsafe fn attach_gl_render(
        &self,
        get_proc_address: ProcAddressFn,
        options: GlRenderOptions,
        on_update: impl Fn() + Send + Sync + 'static,
    ) -> Result<()> {
        let _attaching = self.attach.lock();
        if self.render.lock().is_some() {
            return Err(Error::AlreadyAttached);
        }
        // Construct with the render lock *released*: registration fires
        // `on_update` synchronously, and holding the render lock across
        // that call would deadlock an `on_update` that touches render
        // methods. The attach lock keeps a second attacher out, so the
        // slot check above stays authoritative. (An Arc clone goes in —
        // never the engine's own reference — so a failed create can't
        // drop the core.)
        // SAFETY: GL-currency contract forwarded to the caller (above).
        let render =
            unsafe { GlRender::create(self.mpv.clone(), get_proc_address, options, on_update)? };
        *self.render.lock() = Some(RenderBackend::Gl(render));
        tracing::debug!("mpv GL render context attached");
        Ok(())
    }

    /// Create the software render context: frames arrive as RGBA bytes
    /// via [`render_sw`](Self::render_sw), no GL anywhere — the
    /// backend for shells that upload pixels themselves (or hand them to
    /// a non-GL compositor). Unlike the GL backend there are no
    /// context-current requirements, for rendering or teardown.
    ///
    /// `on_update` has the same contract as in
    /// [`attach_gl_render`](Self::attach_gl_render): fires on mpv's
    /// render thread plus once synchronously (outside the render lock,
    /// before the context is stored), and typically captures a `Weak`
    /// handle — construct, share, attach, then load.
    pub fn attach_sw_render(&self, on_update: impl Fn() + Send + Sync + 'static) -> Result<()> {
        // Same locking shape as `attach_gl_render`, for the same reasons.
        let _attaching = self.attach.lock();
        if self.render.lock().is_some() {
            return Err(Error::AlreadyAttached);
        }
        let render = SwRender::create(self.mpv.clone(), on_update)?;
        *self.render.lock() = Some(RenderBackend::Sw(render));
        tracing::debug!("mpv software render context attached");
        Ok(())
    }

    /// Drop the render context *now*. For the OpenGL backend, call with
    /// the GL context still current (GTK: from the `unrealize` handler):
    /// freeing without the right context current leaks mpv's GL objects
    /// into whatever context is current — in GTK that painted artifacts
    /// over the whole window. The software backend has no such
    /// requirement; detach from any thread.
    pub fn detach_render(&self) {
        if self.render.lock().take().is_some() {
            tracing::debug!("mpv render context detached");
        }
    }

    /// Whether a render context (of either backend) is currently attached.
    pub fn has_render(&self) -> bool {
        self.render.lock().is_some()
    }

    /// Process pending render work after an update callback fired (never
    /// call it from inside the callback itself — that's forbidden, like
    /// any other engine call there). Returns `true` when a new frame
    /// should be drawn. Optional under default options; **mandatory
    /// promptly after every update callback** when the GL backend was
    /// attached with [`GlRenderOptions::advanced_control`]. `false` when
    /// no backend is attached. For the GL backend, the attach contract's
    /// GL-currency rule covers this call too.
    pub fn render_update(&self) -> bool {
        self.render
            .lock()
            .as_mut()
            .is_some_and(RenderBackend::update)
    }

    /// Draw the current frame into `fbo` (`0` = default framebuffer) with
    /// the GL context current. No-op before
    /// [`attach_gl_render`](Self::attach_gl_render); errors with
    /// [`Error::RenderBackendMismatch`] if the software backend is
    /// attached instead. `flip_y` flips the output for flipped-origin
    /// targets (GTK's GLArea wants `true`). Whether this call blocks
    /// until the frame's target display time was fixed at attach
    /// ([`GlRenderOptions::block_for_target_time`]; the default blocks).
    pub fn render_gl(&self, fbo: i32, w: i32, h: i32, flip_y: bool) -> Result<()> {
        match self.render.lock().as_mut() {
            Some(RenderBackend::Gl(r)) => r.render(fbo, w, h, flip_y),
            Some(_) => Err(Error::RenderBackendMismatch),
            None => Ok(()),
        }
    }

    /// Render the current frame as RGBA8 into `buf` (resized to
    /// `w * h * 4`; alpha always opaque). Callable from any thread.
    /// No-op before [`attach_sw_render`](Self::attach_sw_render) —
    /// `buf` is left untouched; errors with
    /// [`Error::RenderBackendMismatch`] if the OpenGL backend is
    /// attached instead.
    pub fn render_sw(&self, w: i32, h: i32, buf: &mut Vec<u8>) -> Result<()> {
        match self.render.lock().as_mut() {
            Some(RenderBackend::Sw(r)) => r.render(w, h, buf),
            Some(_) => Err(Error::RenderBackendMismatch),
            None => Ok(()),
        }
    }
}

// No manual `Drop`: the render context co-owns the core through its own
// `Arc<Mpv>`, so it is structurally freed before the core terminates —
// field order can't break that anymore. If a GL target was attached,
// prefer an explicit `detach_render` with the GL context current before
// dropping; the implicit drop can't make that guarantee. (Wakeup- and
// update-callback teardown is rsmpv's: closures are released safely even
// against in-flight invocations, so mpv can never fire into freed memory.)
