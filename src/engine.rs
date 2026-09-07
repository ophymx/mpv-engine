use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};

use parking_lot::Mutex;
use rsmpv::{EndFileReason, Event, Format, Mpv, PropertyData, sys};

use crate::error::{Error, Result, describe_code};
#[cfg(export_backend)]
use crate::export::{ExportOptions, ExportedFrame, ExportedRender};
use crate::render::{
    GlRender, GlRenderOptions, ProcAddressFn, RenderBackend, RenderKind, SwRender,
};

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

/// The shell's render-update callback as held in the engine's relay
/// slot (see [`Engine::render_update_cb`]).
type UpdateCallback = Arc<dyn Fn() + Send + Sync>;

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
            #[cfg(export_backend)]
            orphaned_render: Mutex::new(None),
            pump: Mutex::new(()),
            next_observe_id: AtomicU64::new(1),
            pending_load: Mutex::new(None),
            deferred_events: Mutex::new(Vec::new()),
            render_update_cb: Arc::new(Mutex::new(Arc::new(|| {}) as UpdateCallback)),
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
    /// `JoinHandle` of an export render thread whose detach ran *on*
    /// that thread (a shell calling
    /// [`detach_render`](Engine::detach_render) from `on_update`): the
    /// self path can't join, so the handle parks here and the next
    /// attach / detach / engine drop on another thread joins the
    /// deferred teardown — instead of leaving the thread's GL/mpv
    /// teardown to race process exit forever.
    #[cfg(export_backend)]
    orphaned_render: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Keeps each [`pump_events`](Engine::pump_events) drain atomic.
    /// rsmpv's `poll_event` is safe to call concurrently, but concurrent
    /// pollers *split* the stream (each event goes to exactly one
    /// caller) — two racing pumps would tear ordered sequences like
    /// `Loaded` → `Ended` across their result batches.
    pump: Mutex<()>,
    /// Userdata ids handed to `mpv_observe_property`; each observation
    /// gets a fresh one so [`unobserve`](Engine::unobserve) is precise.
    next_observe_id: AtomicU64,
    /// Source queued by [`load_when_ready`](Engine::load_when_ready)
    /// until a render context attaches; the attach methods take it and
    /// issue the `loadfile`. Cleared by any transport call that decides
    /// what plays — `load`/`load_paused`/`stop`, or their spellings
    /// through [`command`](Engine::command) — the newest one wins.
    ///
    /// Doubles as the transport-ordering lock: every such call mutates
    /// the slot and issues its mpv command *under this guard*, so
    /// "newest wins" is real under concurrency — a drain can't interleave
    /// with a `stop` between taking the slot and the `loadfile` landing.
    /// Leaf lock: nothing acquired while it is held (mpv commands take no
    /// engine locks; wakeup callbacks are documented to call no engine
    /// methods).
    pending_load: Mutex<Option<String>>,
    /// Engine-synthesized events prepended by
    /// [`pump_events`](Engine::pump_events) — currently only a `Failed`
    /// when a deferred load errors at attach time, so the attach methods'
    /// `Err` can keep meaning "no context was attached".
    deferred_events: Mutex<Vec<PlaybackEvent>>,
    /// The live render-update callback, behind the relay closure that is
    /// what actually gets registered with rsmpv at attach.
    /// [`set_render_update_callback`](Engine::set_render_update_callback)
    /// swaps this slot instead of re-registering through FFI, so the
    /// swap holds no engine lock across a callback invocation — the
    /// relay clones the `Arc` out under a short lock and invokes with
    /// the lock released. All mutation goes through
    /// [`set_update_slot`](Engine::set_update_slot), which drops the
    /// displaced closure outside the lock.
    render_update_cb: Arc<Mutex<UpdateCallback>>,
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
    /// For video engines, prefer [`load_when_ready`](Self::load_when_ready)
    /// until the shell's surface is mapped: `loadfile` before a render
    /// context exists fails VO init and **drops the video track** (see
    /// `load_when_ready`'s docs for the full failure). Loading paused
    /// doesn't dodge it — the load itself is what fails — which is why
    /// the deferred variant exists and why
    /// [`load_paused`](Self::load_paused) is no pre-attach alternative.
    pub fn load(&self, source: &str) -> Result<()> {
        // A plain load supersedes any pending deferred load; the guard is
        // held across the command so the two are one transport step.
        let mut pending = self.pending_load.lock();
        *pending = None;
        self.loadfile(source)
    }

    /// Issue the raw `loadfile` command, touching no engine state. The
    /// callers own the [`pending_load`](Self::pending_load) transport
    /// step around this.
    fn loadfile(&self, source: &str) -> Result<()> {
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

    /// [`load`](Self::load), deferred until frames have somewhere to go:
    /// on a render-API engine (`vo=libmpv`, [`Engine::video`]) with no
    /// context attached yet, the source is queued and the attach call
    /// ([`attach_gl_render`](Self::attach_gl_render) /
    /// [`attach_sw_render`](Self::attach_sw_render)) issues the
    /// `loadfile` — the ordering a video shell wants, without
    /// hand-carrying a pending-source slot between its load path and its
    /// realize handler. On an engine that is already attached — or whose
    /// `vo` never uses the render API ([`headless`](Self::headless), a
    /// windowed vo), so no attach is coming — this is plain
    /// [`load`](Self::load).
    ///
    /// The whole `loadfile` is deferred, not just an unpause, because a
    /// load before the render context exists doesn't merely start
    /// blind: mpv fails to initialize the video output and **drops the
    /// video track** — a video-only file dies with
    /// `MPV_ERROR_NOTHING_TO_PLAY` (-16) even when loaded paused, and a
    /// file with audio plays sound over a permanently black surface.
    ///
    /// The queued source is a *pending intent*: a later
    /// [`load`](Self::load), [`load_paused`](Self::load_paused), or
    /// [`stop`](Self::stop) before the attach supersedes it (newest
    /// transport call wins — including the same commands issued through
    /// [`command`](Self::command)), and a second `load_when_ready`
    /// replaces it. Pause
    /// state needs no special casing — the `pause` property persists
    /// across `loadfile`, so a consumer that pauses before the attach
    /// gets the deferred file loaded paused, exactly as if it had been
    /// playing.
    ///
    /// The defer-or-load decision reads the **current** `vo` property,
    /// so it tracks runtime `vo` changes (via
    /// [`set_property`](Self::set_property)) and values picked up from a
    /// config file — not just what the builder set. The same rule covers
    /// the window after a [`detach_render`](Self::detach_render): with
    /// `vo` still on the render API, sources queue again awaiting a
    /// re-attach — a shell going render-less for good should switch `vo`
    /// (e.g. to `null`) so loads run immediately. A deferred `loadfile`
    /// that fails at attach time surfaces as
    /// [`PlaybackEvent::Failed`] on the next
    /// [`pump_events`](Self::pump_events), never as an `Err` from the
    /// attach call.
    pub fn load_when_ready(&self, source: &str) -> Result<()> {
        if !self.expects_render() || self.has_render() {
            return self.load(source);
        }
        *self.pending_load.lock() = Some(source.to_owned());
        // An attach can slip in between `has_render()` above and the
        // queueing; it would find the slot empty and never load. Re-check
        // and drain — the `take` inside makes exactly one loader win if
        // the attach also saw the source.
        if self.has_render() {
            return self.load_pending();
        }
        Ok(())
    }

    /// Whether frames route through the render API right now: the
    /// current `vo` property includes `libmpv` (it accepts a
    /// comma-separated fallback chain). Read live rather than snapshotted
    /// at build — `vo` is runtime-settable through this crate's own
    /// [`set_property`](Self::set_property), and a config file can set it
    /// behind the builder's back. On a read failure, err toward a plain
    /// load: the `loadfile` then surfaces the real error instead of the
    /// source silently parking in the queue.
    fn expects_render(&self) -> bool {
        self.get_property::<String>("vo")
            .is_ok_and(|vo| vo.split(',').any(|part| part.trim() == "libmpv"))
    }

    /// Escape hatch: any mpv command, args passed as an array (no quoting
    /// needed).
    ///
    /// Commands that decide what plays next — `loadfile`, `loadlist`,
    /// `stop`, `quit`, `quit-watch-later` — also discard a load queued by
    /// [`load_when_ready`](Self::load_when_ready), same as the typed
    /// transport methods: this is the only way to issue `loadfile` with
    /// flags, and a superseded source must not resurface at attach time.
    pub fn command(&self, name: &str, args: &[&str]) -> Result<()> {
        let mut argv = Vec::with_capacity(args.len() + 1);
        argv.push(name);
        argv.extend_from_slice(args);
        let supersedes_pending = matches!(
            name,
            "loadfile" | "loadlist" | "stop" | "quit" | "quit-watch-later"
        );
        // Transport commands run under the pending_load guard (clear +
        // command as one step); everything else goes straight through.
        let _transport = supersedes_pending.then(|| {
            let mut pending = self.pending_load.lock();
            *pending = None;
            pending
        });
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
    /// [`PlaybackEvent::Ended`] with [`EndReason::Stop`]. Also discards a
    /// load queued by [`load_when_ready`](Self::load_when_ready) — there
    /// is nothing left to play.
    pub fn stop(&self) -> Result<()> {
        // `command` clears the pending deferred load (transport command).
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
        // Engine-synthesized events first (a deferred load that failed at
        // attach time) — they predate whatever mpv has queued now.
        let mut out = std::mem::take(&mut *self.deferred_events.lock());
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
    /// A successful attach also issues any load queued by
    /// [`load_when_ready`](Self::load_when_ready). `Err` still means "no
    /// context was attached" — a deferred `loadfile` that fails here
    /// leaves the context in place and surfaces as
    /// [`PlaybackEvent::Failed`] on the next
    /// [`pump_events`](Self::pump_events) (the wakeup callback fires).
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
        // Join any teardown deferred by a detach-from-`on_update` before
        // creating a context: the orphaned thread frees the old mpv
        // render context on exit, and libmpv allows only one per core.
        #[cfg(export_backend)]
        self.join_orphaned_render();
        if self.render.lock().is_some() {
            return Err(Error::AlreadyAttached);
        }
        // Construct with the render lock *released*: registration fires
        // `on_update` synchronously (via the relay), and holding the
        // render lock across that call would deadlock an `on_update` that
        // touches render methods. The attach lock keeps a second attacher
        // out, so the slot check above stays authoritative. (An Arc clone
        // goes in — never the engine's own reference — so a failed create
        // can't drop the core.)
        self.set_update_slot(Arc::new(on_update));
        // SAFETY: GL-currency contract forwarded to the caller (above).
        let created = unsafe {
            GlRender::create(
                self.mpv.clone(),
                get_proc_address,
                options,
                self.update_relay(),
            )
        };
        let render = match created {
            Ok(r) => r,
            Err(e) => {
                // Failed attach: don't pin the shell closure's captures
                // in a slot nothing will ever fire.
                self.set_update_slot(Arc::new(|| {}));
                return Err(e);
            }
        };
        *self.render.lock() = Some(RenderBackend::Gl(render));
        tracing::debug!("mpv GL render context attached");
        self.drain_pending_load();
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
    ///
    /// As with [`attach_gl_render`](Self::attach_gl_render): a successful
    /// attach issues any [`load_when_ready`](Self::load_when_ready)
    /// queue, `Err` still means "no context was attached", and a deferred
    /// `loadfile` failing here surfaces as [`PlaybackEvent::Failed`] on
    /// the next [`pump_events`](Self::pump_events) instead.
    pub fn attach_sw_render(&self, on_update: impl Fn() + Send + Sync + 'static) -> Result<()> {
        // Same locking shape as `attach_gl_render`, for the same reasons.
        let _attaching = self.attach.lock();
        #[cfg(export_backend)]
        self.join_orphaned_render();
        if self.render.lock().is_some() {
            return Err(Error::AlreadyAttached);
        }
        self.set_update_slot(Arc::new(on_update));
        let render = match SwRender::create(self.mpv.clone(), self.update_relay()) {
            Ok(r) => r,
            Err(e) => {
                self.set_update_slot(Arc::new(|| {}));
                return Err(e);
            }
        };
        *self.render.lock() = Some(RenderBackend::Sw(render));
        tracing::debug!("mpv software render context attached");
        self.drain_pending_load();
        Ok(())
    }

    /// Issue the `loadfile` a [`load_when_ready`](Self::load_when_ready)
    /// queued, now that a render context is stored. The guard is held
    /// across the `loadfile` — take-then-load must be one transport step,
    /// or a concurrent `stop`/`load` landing in between would be
    /// overridden by the older, superseded source.
    fn load_pending(&self) -> Result<()> {
        let mut pending = self.pending_load.lock();
        match pending.take() {
            Some(source) => self.loadfile(&source),
            None => Ok(()),
        }
    }

    /// [`load_pending`](Self::load_pending) for the attach paths, where
    /// an `Err` must keep meaning "no context was attached": the freshly
    /// stored context stays either way, so a deferred-load failure is
    /// rerouted onto the event stream as [`PlaybackEvent::Failed`] — the
    /// channel an asynchronous load failure was headed for anyway — and
    /// the wakeup callback is tickled so push-driven shells pump for it.
    fn drain_pending_load(&self) {
        let Err(err) = self.load_pending() else {
            return;
        };
        tracing::warn!("deferred load failed at attach: {err}");
        let code = match &err {
            Error::Mpv(e) => e.raw_code().unwrap_or(sys::MPV_ERROR_GENERIC),
            _ => sys::MPV_ERROR_GENERIC,
        };
        self.deferred_events.lock().push(PlaybackEvent::Failed {
            code,
            message: describe_code(code),
        });
        self.mpv.wakeup();
    }

    /// Drop the render context *now*. For the OpenGL backend, call with
    /// the GL context still current (GTK: from the `unrealize` handler):
    /// freeing without the right context current leaks mpv's GL objects
    /// into whatever context is current — in GTK that painted artifacts
    /// over the whole window. The software backend has no such
    /// requirement; detach from any thread.
    ///
    /// After a detach, [`load_when_ready`](Self::load_when_ready) defers
    /// again — `vo` still names the render API, so sources queue awaiting
    /// a re-attach. A shell detaching *for good* (say, dropping to
    /// audio-only) should also switch `vo` (e.g.
    /// [`set_property`](Self::set_property)`("vo", "null")`); the
    /// defer-or-load decision reads the live `vo`, so loads then run
    /// immediately instead of parking.
    pub fn detach_render(&self) {
        // Locking here answers three hazards at once:
        //
        // * The take + callback-slot reset happen inside one render-lock
        //   scope, so a racing attach — which re-checks the slot under
        //   the render lock before registering its `on_update` — orders
        //   its registration strictly after our reset. (Resetting after
        //   the take, unordered, could clobber a concurrently-completed
        //   attach's fresh callback: a silent permanent freeze.)
        // * Off the render thread (the normal case), the attach lock is
        //   held across take/reset/drop, so an attach can't call
        //   `mpv_render_context_create` while the old context is still
        //   being freed — libmpv allows one render context per core, and
        //   that race read as a spurious attach failure.
        // * The backend itself still drops with the render lock
        //   released: the exported backend's drop joins its render
        //   thread, and an in-flight frame-published callback calling
        //   `acquire_frame` (render lock) would deadlock against a join
        //   performed under it.
        //
        // The exception is a detach from the exported backend's *own*
        // render thread (a shell tearing down from `on_update`): taking
        // the attach lock there can deadlock against a concurrent detach
        // that holds it while joining this very thread — so the self
        // path skips the attach lock (its take/reset ordering still
        // holds via the render lock), and `ExportedRender`'s drop skips
        // the self-join, deferring thread exit to just after the
        // callback returns — with the thread's handle parked in
        // `orphaned_render` so a later attach/detach/engine-drop joins
        // the deferred teardown instead of racing it.
        let on_render_thread = {
            let slot = self.render.lock();
            match slot.as_ref() {
                // Nothing attached: done. Returning without touching the
                // attach lock also lets a callback-detach racing a
                // concurrent detach (which already took the backend and
                // now joins this thread under the attach lock) unwind
                // instead of deadlocking.
                None => return,
                Some(backend) => backend.on_own_render_thread(),
            }
        };
        let _attaching = if on_render_thread {
            None
        } else {
            let attaching = self.attach.lock();
            // A prior detach-from-callback may have parked its render
            // thread; join that deferred teardown first (under the
            // attach lock, so it is fully over before this detach's own
            // backend drop and any subsequent attach).
            #[cfg(export_backend)]
            self.join_orphaned_render();
            Some(attaching)
        };
        let (taken, displaced) = {
            let mut slot = self.render.lock();
            #[cfg_attr(not(export_backend), allow(unused_mut))]
            let mut taken = slot.take();
            // Self path: the backend's drop below can't join its own
            // thread, so park the handle for the next engine call from
            // another thread to join. Parked *inside* this render-lock
            // scope, so once the slot reads empty the handle is already
            // there — an attach that saw the slot clear joins it rather
            // than racing the deferred teardown. (A still-parked older
            // orphan, if any, has long exited and drops detached —
            // exactly its pre-parking fate.)
            #[cfg(export_backend)]
            if on_render_thread {
                if let Some(RenderBackend::Exported(render)) = taken.as_mut() {
                    *self.orphaned_render.lock() = render.take_thread();
                }
            }
            // Swap the shell's callback out with the context: nothing
            // can fire it anymore, and keeping it would pin its captures
            // (typically shell window state) until the next attach. Only
            // the *swap* happens under the render lock (ordering, per
            // above); the displaced closure drops at the end of this fn,
            // outside every engine lock — its captures' `Drop` may call
            // back into the engine.
            let displaced = taken.as_ref().map(|_| {
                std::mem::replace(
                    &mut *self.render_update_cb.lock(),
                    Arc::new(|| {}) as UpdateCallback,
                )
            });
            (taken, displaced)
        };
        if taken.is_some() {
            // The backend drops while the attach lock (when held) is
            // still ours: this joins the export render thread and frees
            // the mpv render context, which must finish before a
            // concurrent attach may create a new context.
            drop(taken);
            tracing::debug!("mpv render context detached");
        }
        drop(_attaching);
        // Only now, with no engine lock held, release the shell's
        // callback: its captures' `Drop` may call back into the engine —
        // attach methods included, which take the attach lock.
        drop(displaced);
    }

    /// Join a render thread parked by a detach-from-`on_update` (see
    /// `orphaned_render`). No-op when nothing is parked; when called
    /// *on* the orphaned thread itself (another engine call from that
    /// same callback), the handle stays parked for a real joiner —
    /// self-joining would deadlock.
    #[cfg(export_backend)]
    fn join_orphaned_render(&self) {
        let mut parked = self.orphaned_render.lock();
        if let Some(handle) = parked.take() {
            if handle.thread().id() == std::thread::current().id() {
                *parked = Some(handle);
            } else {
                drop(parked);
                let _ = handle.join();
            }
        }
    }

    /// Whether a render context (of either backend) is currently
    /// attached. Delegates to [`attached_render`](Self::attached_render)
    /// — one read of the slot, so the two can never disagree.
    pub fn has_render(&self) -> bool {
        self.attached_render().is_some()
    }

    /// Which render backend is attached, if any — the "which one"
    /// companion to [`has_render`](Self::has_render), for shells that
    /// route between per-backend code paths (say, GPU texture sampling
    /// vs. RGBA upload) without having to track the attach outcome in
    /// state of their own.
    pub fn attached_render(&self) -> Option<RenderKind> {
        self.render.lock().as_ref().map(RenderBackend::kind)
    }

    /// Store `cb` as the live render-update callback, dropping the
    /// displaced closure *outside* the slot lock — its captures may carry
    /// a `Drop` that calls back into the engine, which must not run under
    /// any engine lock.
    fn set_update_slot(&self, cb: UpdateCallback) {
        let old = std::mem::replace(&mut *self.render_update_cb.lock(), cb);
        drop(old);
    }

    /// The closure actually registered with rsmpv at attach: reads the
    /// engine's callback slot on every fire, so
    /// [`set_render_update_callback`](Self::set_render_update_callback)
    /// can swap the target without re-registering through FFI. The `Arc`
    /// is cloned out under a short lock and invoked with the lock
    /// released — no engine lock is ever held across a callback
    /// invocation.
    fn update_relay(&self) -> impl Fn() + Send + Sync + 'static {
        let slot = Arc::clone(&self.render_update_cb);
        move || {
            let cb = Arc::clone(&*slot.lock());
            cb();
        }
    }

    /// Replace the render-update callback registered at attach — the same
    /// post-registration replaceability
    /// [`set_wakeup_callback`](Self::set_wakeup_callback) has, for the
    /// render seam. For shells that can only build their real closure
    /// after the engine is shared: attach with a placeholder, wrap the
    /// engine in your `Arc`/shared structure, then register the
    /// weak-capturing closure here.
    ///
    /// The new callback takes over the attach-time contract: it fires on
    /// mpv's render thread — and **once synchronously on the calling
    /// thread, from inside this very call** (registration raises an
    /// update immediately, so a frame signaled to the old callback isn't
    /// lost). The synchronous fire runs outside every engine lock, same
    /// as at attach — an engine call from inside it cannot deadlock. The
    /// standing rule still applies to the mpv-thread fires, though: do no
    /// work and call no engine methods inside — signal your main loop and
    /// render/pump from there.
    ///
    /// The replaced closure is released with no engine lock held: on this
    /// thread during this call when no invocation is in flight, otherwise
    /// when its last in-flight invocation finishes — possibly on an
    /// mpv-internal thread, so captures whose `Drop` calls into libmpv
    /// (e.g. a last `Engine`-owning handle) don't belong in an update
    /// callback.
    ///
    /// The registration is tied to the attached context:
    /// [`detach_render`](Self::detach_render) releases it, and the next
    /// attach starts from that attach's own `on_update`.
    ///
    /// Errors with [`Error::NotAttached`] when no render context is
    /// attached — a callback that could never fire is a wiring bug,
    /// surfaced loudly rather than silently dropped.
    pub fn set_render_update_callback(
        &self,
        on_update: impl Fn() + Send + Sync + 'static,
    ) -> Result<()> {
        if !self.has_render() {
            return Err(Error::NotAttached);
        }
        let new: UpdateCallback = Arc::new(on_update);
        self.set_update_slot(Arc::clone(&new));
        // The synchronous registration fire, with no engine lock held.
        new();
        Ok(())
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

    /// Create the exported-frame render backend (`export` feature): the
    /// engine spawns a render thread owning a hidden GL context, mpv
    /// renders there into exportable framebuffers (IOSurface-backed on
    /// macOS, DMA-BUF-backed on Linux, shared-D3D11-texture-backed on
    /// Windows), and the shell pulls zero-copy [`ExportedFrame`]s with
    /// [`acquire_frame`](Self::acquire_frame) to import into Metal /
    /// Vulkan / D3D / wgpu.
    /// Fully safe — no GL context or currency contract crosses this API;
    /// the thread that creates the context is the thread that renders on
    /// it and frees it.
    ///
    /// `on_update` differs from the other backends' callback: it fires
    /// **after a frame is published**, from the engine's own render
    /// thread (plus the usual synchronous fire when
    /// [`set_render_update_callback`](Self::set_render_update_callback)
    /// replaces it, and once from inside a successful attach when a
    /// frame was already published while the attach was completing — so
    /// a poke is never lost to that window). Calling
    /// [`acquire_frame`](Self::acquire_frame)
    /// inside it is fine; just don't block in it — it stalls video
    /// pacing. Frames published before the shell drains them are
    /// replaced, newest wins.
    ///
    /// Tearing down from inside `on_update` —
    /// [`detach_render`](Self::detach_render), or dropping the last
    /// engine handle — is supported: the render thread's teardown is
    /// deferred to just after the callback returns instead of joined
    /// (which would self-deadlock). A `detach_render` parks the deferred
    /// thread's handle, and the next attach/detach/engine-drop from
    /// another thread joins it — so a re-attach orders strictly after
    /// the old teardown rather than racing it. One limit applies:
    /// **attach calls must not be made from inside `on_update`** (nor
    /// from callback captures' `Drop`) — a concurrent detach may hold
    /// the attach serialization lock while waiting on this very thread.
    ///
    /// Shares the single render slot with the other backends
    /// ([`Error::AlreadyAttached`]); [`detach_render`](Self::detach_render)
    /// shuts the render thread down (no GL-currency obligation for the
    /// caller — unique among the GL-based backends), and a successful
    /// attach issues any [`load_when_ready`](Self::load_when_ready)
    /// queue, with the same failure routing as the other attach methods.
    ///
    /// Errors with [`Error::ExportSetup`] when no GL context can be
    /// created — typically a session without GPU access (no
    /// WindowServer on macOS, no readable DRM render node on Linux, no
    /// OpenGL ICD or no `WGL_NV_DX_interop2` on Windows); treat it like
    /// a missing display.
    #[cfg(export_backend)]
    pub fn attach_exported_render(
        &self,
        options: ExportOptions,
        on_update: impl Fn() + Send + Sync + 'static,
    ) -> Result<()> {
        // Same locking shape as `attach_gl_render`, for the same reasons.
        let attaching = self.attach.lock();
        // A detach-from-`on_update` defers its render thread's teardown;
        // join it here so this attach orders strictly after it (and so
        // its `Error::ExportSetup` window can't be hit by this path).
        self.join_orphaned_render();
        if self.render.lock().is_some() {
            return Err(Error::AlreadyAttached);
        }
        self.set_update_slot(Arc::new(on_update));
        let created = ExportedRender::create(self.mpv.clone(), options, self.update_relay());
        let render = match created {
            Ok(r) => r,
            Err(e) => {
                self.set_update_slot(Arc::new(|| {}));
                return Err(e);
            }
        };
        let shared = Arc::clone(render.shared());
        *self.render.lock() = Some(RenderBackend::Exported(render));
        tracing::debug!("mpv exported render context attached");
        self.drain_pending_load();
        drop(attaching);
        // A frame published between the render thread coming up and the
        // backend landing in the slot fired `on_update` into a window
        // where `acquire_frame` still read an empty slot. Re-fire once
        // now that the frame is acquirable — after every engine lock is
        // released, per the callback contract.
        if shared.has_published() {
            (self.update_relay())();
        }
        Ok(())
    }

    /// Take the newest published frame from the exported backend, if one
    /// is waiting. `Ok(None)` both when no frame has been published since
    /// the last acquire and when no backend is attached (ordinary startup
    /// state, mirroring [`render_gl`](Self::render_gl)'s no-op);
    /// [`Error::RenderBackendMismatch`] when a different backend is
    /// attached. Callable from any thread, including from inside the
    /// exported backend's `on_update`.
    #[cfg(export_backend)]
    pub fn acquire_frame(&self) -> Result<Option<ExportedFrame>> {
        // Clone the shared state out under a short slot lock; the take
        // itself must not hold the render lock (an attach/detach could
        // block behind an unrelated pool operation otherwise).
        let shared = match self.render.lock().as_ref() {
            Some(RenderBackend::Exported(r)) => Arc::clone(r.shared()),
            Some(_) => return Err(Error::RenderBackendMismatch),
            None => return Ok(None),
        };
        Ok(ExportedFrame::take_published(shared))
    }

    /// Resize the exported backend's frames: takes effect from the next
    /// rendered frame (which is forced promptly, so a paused or ended
    /// video re-renders at the new size instead of waiting for playback
    /// to produce one — and if the shell happens to be holding every
    /// pool buffer at that moment, the forced render re-arms as soon as
    /// a frame handle is released, rather than being lost). Zero in
    /// either dimension pauses rendering until a
    /// real size arrives — map-before-layout states in a shell.
    /// No-op `Ok` when nothing is attached;
    /// [`Error::RenderBackendMismatch`] for a different backend.
    /// Outstanding [`ExportedFrame`]s keep their original size.
    #[cfg(export_backend)]
    pub fn set_export_size(&self, width: u32, height: u32) -> Result<()> {
        match self.render.lock().as_ref() {
            Some(RenderBackend::Exported(r)) => {
                r.shared().set_target_size(width, height);
                Ok(())
            }
            Some(_) => Err(Error::RenderBackendMismatch),
            None => Ok(()),
        }
    }
}

// No resource-management `Drop`: the render context co-owns the core
// through its own `Arc<Mpv>`, so it is structurally freed before the core
// terminates — field order can't break that anymore. If a GL target was
// attached, prefer an explicit `detach_render` with the GL context current
// before dropping; the implicit drop can't make that guarantee. (Wakeup-
// and update-callback teardown is rsmpv's: closures are released safely
// even against in-flight invocations, so mpv can never fire into freed
// memory.) The `export` Drop below only joins a parked render thread —
// it manages no resource the field drops don't already cover.

#[cfg(export_backend)]
impl Drop for Engine {
    fn drop(&mut self) {
        // A detach-from-`on_update` parks its render thread's handle
        // (see `orphaned_render`); join it so the deferred GL/mpv
        // teardown can't race process exit. When the engine itself dies
        // on that thread, `join_orphaned_render` leaves the handle
        // parked and it drops detached — same as before parking existed.
        self.join_orphaned_render();
    }
}
