# mpv-engine

Toolkit-agnostic core for embedding libmpv in Rust applications: handle
lifecycle, commands, properties, typed playback events, and render seams.

Extracted from two independent embedders after their engine layers
converged on the same API without coordinating:

- a **GTK4 app** that renders mpv directly into a `GtkGLArea`'s FBO
  inside GTK's GL context (the GNOME Celluloid pattern), and
- an **iced widget** that renders into its own side EGL context, then
  reads back RGBA (SW path) or exports the FBO to a `wgpu::Texture`
  (GPU path).

Those two render strategies are *legitimately different* — the shared part
is everything underneath, and that's this crate.

## What lives here (and the rule that decides it)

A quirk belongs in `mpv-engine` if its **consumer-facing interface
survives the upstream fix**. Anything whose shape would change with a
toolkit's API belongs in that toolkit's adapter crate.

Owned here today:

- `LC_NUMERIC=C` forced before `mpv_create` — toolkits (GTK included)
  call `setlocale()` from the environment; mpv's number parsing breaks
  under comma-decimal locales. Both parent projects carried this guard
  independently.
- Render-context lifecycle: created only when a GL context is live, freed
  strictly **before** the mpv handle, and freed **with the GL context
  current** (`detach_render`) — otherwise mpv's GL objects leak into
  whatever context is current (in GTK: whole-window artifacts after
  popping the player page).
- Commands as pre-tokenized argument arrays (`mpv_command`), never a
  joined command string — paths with spaces/quotes need no escaping.
  Pinned by a regression test (`loadfile_handles_awkward_filenames`)
  because the string-joined variant shipped a real bug.
- Typed events: `pump_events()` → `Loaded` / `Ended{reason}` (EOF vs
  stop, for playlist logic) / `PlaybackRestart` (seek finished — snap
  scrubbers here) / `PropertyChanged` / `Failed{code, message}`,
  with mpv's negative error codes expanded to `mpv_error_string` text
  plus the numeric code (diagnostic strings — user-facing copy is the
  integrator's job). Errored
  end-of-file arrives as `Failed`, not silently.
- The update callback's threading contract: fires on mpv's render thread
  (`Send` required), **plus once synchronously at registration**; the
  crate has no main-loop opinion — bridging to GTK's `spawn_local` or
  iced's subscriptions is the shell's job.
- `load_paused()` — pause set *before* `loadfile`, so demuxing doesn't
  start before the shell's window is mapped (the init-time variant of the
  same idea tends to hang).
- A software render backend (`attach_sw_render`/`render_sw`, RGBA into a
  caller buffer, no GL) sharing the one attach slot with the GL backend —
  including the fix for mpv's `"rgb0"` output leaving the fourth byte
  undefined, which consumers treating the buffer as RGBA read as garbage
  alpha.
- An event wakeup seam (`set_wakeup_callback`) so shells get a push
  signal when events queue instead of polling `pump_events` on a timer —
  the only timely path for audio-only use or failures while paused.
- Property access through crate-owned types (`PropertyValue`, the sealed
  `PropertyGet`) — the binding's conversion traits never appear in this
  crate's public API, so a binding major bump can't become a semver break
  here.

Explicitly **not** here: widgets, main-loop integration, toolkit types,
and git-pinned dependencies (crates.io forbids them, and they'd re-couple
this crate to exactly the churn it exists to contain).

## Consumers

```text
mpv-engine (core: this crate)
├── GTK adapter    widget/controls over core (GTK GLArea → render_gl)
└── iced adapter   widget + Subscription + shader pipeline; side-context render
```

Shells attach a render target with `attach_gl_render(get_proc_address,
options, on_update)` (an `unsafe` fn — it carries the GL-context-currency
contract the type system can't express: the target context must be
current at attach, every `render_gl`/`render_update`, and detach/drop)
and draw via `render_gl(fbo, w, h, flip_y)` from their paint handler —
or, GL-free and fully safe, `attach_sw_render(on_update)` and pull RGBA
bytes with `render_sw(w, h, &mut buf)`. `GlRenderOptions` fixes the
shell's render-loop discipline at attach: the default blocks `render_gl`
until the frame's target display time (right for a GTK paint handler),
while a shell rendering on a compositor thread (iced `prepare`) sets
`block_for_target_time: false` and paces frames itself (or sets
`video-timing-offset=0`); `advanced_control` is there too, obligating
`render_update()` after every update callback. Events are drained with
`pump_events()` on the shell's own cadence, with `set_wakeup_callback`
as the push signal for shells that don't poll. Headless/audio
use (`Engine::headless()`, `vo=null`) needs neither — that's also how the
test suite runs without a display.

On Linux, EGL 1.5's `eglGetProcAddress` resolves everything mpv asks for.
Avoid libepoxy on glvnd builds: it doesn't export core GL symbols as
`dlsym`-able functions and mpv reports `MPV_ERROR_UNSUPPORTED`.

## Roadmap (planned non-default features)

| feature | contents | churn it contains |
|---|---|---|
| `egl` | side EGL context + FBO render + RGBA readback (GL-accelerated; the pure-software path is already in core as `render_sw`) | none — stable APIs |
| `export` | `ExportedFrame`: OPAQUE_FD/DMA-BUF + GL semaphore export (ash only) | the *permanent* workarounds — e.g. wgpu-hal never enables `VK_KHR_external_semaphore_fd`, so the strict-loader `vkGetSemaphoreFdKHR` path lives here indefinitely |
| `wgpu` | import `ExportedFrame` → `wgpu::Texture` via wgpu-hal | wgpu-major lockstep, isolated behind a non-default feature; releases track wgpu majors (the `egui-wgpu` pattern). When iced reaches wgpu 30, `add_wait_semaphore` replaces the empty-submit semaphore hack and `create_texture_from_hal`'s `initial_state` lands — internals change, the feature's API doesn't |

The default feature set never grows unstable dependencies: a consumer on
core + `egl` alone is structurally isolated from all of it.

The iced-side upstream watch (wgpu-30 bump via cryoglyph, damage
tracking, foreign-texture widget, HDR surface config) stays in the iced
adapter crate — those items fail the interface-survival rule.

## Testing

`cargo test` runs headless against real libmpv (`vo=null`); media inputs
are generated with ffmpeg. Both are probed at runtime — missing tooling
skips tests rather than failing them. System deps: the libmpv dev
package to build, ffmpeg to generate test inputs.

## Invariants for contributors

Things a change must not weaken (each is a lesson paid for in one of the
parent projects):

- **Commands are argument arrays**, never a joined command string — no
  matter how convenient `mpv_command_string` looks for a one-off. The
  war story: an earlier command layer joined args into one command
  line, and a filename with spaces produced
  `MPV_ERROR_INVALID_PARAMETER` in production.
  `loadfile_handles_awkward_filenames` pins the behavior, so it also
  guards binding swaps and future major bumps.
- **Drop order**: render context strictly before the mpv handle. Since
  rsmpv 0.2 the ordering is structural — the render context co-owns the
  core via `Arc<Mpv>`, so there is no `Drop` impl to maintain. Don't
  reintroduce a path where a render context can outlive its core.
- **`detach_render` with the GL context current** is the real teardown
  path; `Drop` is only a fallback that can't guarantee it. Don't soften
  that contract in docs or code.
- **Default features stay boring**: no toolkit deps, no git deps
  (crates.io forbids them anyway), no unstable upstreams. Churny interop
  goes behind the non-default features in the roadmap table.

## License

MIT OR Apache-2.0 — and via the clean-room [`rsmpv`] bindings (MIT OR
Apache-2.0 over an ISC sys crate), that now covers the crate's entire
Rust dependency tree.

Binaries still carry libmpv's terms: it is **GPLv2+ in default builds**
(LGPLv2.1+ only when built with `-Dgpl=false`), and FFmpeg's build flags
affect the combined license too. Before distributing anything built on
this crate, see [LICENSING.md](LICENSING.md) for the layer-by-layer
picture and the common gotchas.

[`rsmpv`]: https://crates.io/crates/rsmpv
