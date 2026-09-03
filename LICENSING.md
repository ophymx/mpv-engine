# Licensing: what you actually ship

`mpv-engine`'s own code is **MIT OR Apache-2.0**, and as of the switch to
the [`rsmpv`](https://crates.io/crates/rsmpv) bindings, so is the entire
Rust side of the dependency tree. That answers "may I copy this code?" —
it does **not** answer "what do I owe when I ship a binary?", because
every binary built on this crate dynamically links **libmpv**, and
libmpv's license is what governs distribution. This document maps the
layers so you know where to look.

> **Not legal advice.** This is an engineering summary of the license
> texts and upstream policy as of September 2026 (mpv master, crates.io
> metadata). For a product decision, have counsel read the actual
> licenses: [LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html),
> [GPL-2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html),
> and mpv's [`Copyright`](https://github.com/mpv-player/mpv/blob/master/Copyright).

## The layer cake

```text
your application            — your license
└── mpv-engine              — MIT OR Apache-2.0
    └── rsmpv               — MIT OR Apache-2.0  (safe bindings)
        └── rsmpv-sys       — ISC  (clean-room FFI, written from mpv's
            │                 ISC-licensed client headers)
            └── libmpv.so   — GPLv2+ by default; LGPLv2.1+ only if built
                │             with -Dgpl=false  (dynamically linked)
                └── FFmpeg  — LGPL-2.1+ by default; GPL if built with
                              --enable-gpl; NON-REDISTRIBUTABLE if built
                              with --enable-nonfree
```

Everything above the `libmpv.so` line is permissive. The license of your
shipped binary is decided by the libmpv (and FFmpeg) build you link —
not by the Rust crates.

## The gotchas, in the order they bite

### 1. The permissive crates don't change what libmpv is

MIT/Apache/ISC applies to the Rust code. GPL and LGPL obligations attach
when you **distribute** a binary linking libmpv. Nothing about the
bindings being permissive changes that — the licensing event in this
stack is, and always was, libmpv itself.

### 2. "mpv is LGPL" is only true for a build nobody ships by default

mpv is **GPLv2+** unless configured with `-Dgpl=false`, which produces an
**LGPLv2.1+** libmpv with the remaining GPL-only components (X11 video
output, OSS audio, vdpau, assorted filters) compiled out. Distro packages
are effectively always the full GPL build — they exist to serve the `mpv`
CLI, and the LGPL build is explicitly not recommended for that. **If you
link the libmpv from your distro and distribute the result, plan on GPL
terms**, and note there is no runtime query for the build's license — you
have to read the package's build recipe (Debian `d/rules`, Arch
`PKGBUILD`, your Homebrew formula, etc.) or build libmpv yourself.

Under the FSF's reading, this holds even if you *don't bundle* libmpv and
merely link the system copy: an app written against a GPL library must be
distributed under GPL-compatible terms. That reading is contested, but
it's the conservative baseline.

The silver lining for *this* crate's consumers: the features the LGPL
build drops are mostly mpv's own video outputs — and embedders here use
the render API (`render_gl`/`render_sw`) or `vo=null`, not mpv's VOs. An
LGPL libmpv costs a render-API embedder essentially nothing.

### 3. FFmpeg can silently re-GPL (or worse) an otherwise-LGPL stack

libmpv links FFmpeg. FFmpeg is LGPL-2.1+ **by default**, but:

- `--enable-gpl` (needed for x264, x265, and friends — common in distro
  and "full" builds) makes the combination **GPL**, even under an
  LGPL-built libmpv;
- `--enable-nonfree` (fdk-aac) makes the result
  **non-redistributable** entirely.

If you're building an LGPL stack, you must build FFmpeg too, and audit
its configure flags, not just mpv's.

### 4. Bundling is distributing

You inherit the obligations for everything inside the box you hand out:
an AppImage, a Flatpak with mpv built as a module, a Windows installer
carrying `mpv-2.dll`, a macOS `.app` with a dylib, a Docker image. For
each (L)GPL component you bundle you must ship the license text, an
attribution notice, and the **corresponding source for the exact build
you shipped** (or a written offer for it) — "it's on GitHub" is not
sufficient for a patched or pinned build. Keeping libmpv a separate
dynamic library file (the default with `rsmpv-sys`) is what makes LGPL
§6(b) compliance easy; static-linking libmpv itself into a non-GPL app
gives that path up.

### 5. No distribution, no obligations

GPLv2 and LGPL-2.1 trigger on *distribution*. Internal tools, personal
builds, CI, and server-side use (these licenses have no network clause)
carry no source-sharing obligations at all. Most of this document is
about the moment you hand a binary to someone else.

### 6. App stores add their own constraints

Apple's App Store terms have historically been held incompatible with
GPL (the VLC takedown), and iOS's signed, static distribution model sits
badly with LGPL §6's relinking requirement. The permissive Rust tree
removes the bindings from that equation, but libmpv itself remains
(L)GPL. If an app-store build is in your future, treat licensing as a
project-level decision to make early, with counsel involved.

### 7. License scanners come up clean on the Rust side — mind the gap

`cargo deny check licenses`, FOSSA, Black Duck, etc. should report only
MIT/Apache-2.0/ISC (and similar) for this crate's dependency tree. What
they *can't* see is the system libmpv you link at runtime — gotchas 2–4
still apply and live entirely outside the scanner's view. (And if a
scanner ever flags copyleft *inside* the Rust tree, a dependency
changed under you — audit the lockfile.)

## What this means in practice

| You are shipping… | Effective terms | Your main obligations |
|---|---|---|
| An open-source app (GPL-compatible license), any libmpv | GPLv2+ overall | Notices + source availability — you already comply. (Apache-2.0 app code combines with mpv via the "or later": distribute under GPLv3 terms.) |
| A proprietary app linking the **system/distro** libmpv | GPL, per the conservative reading | Generally not viable as-is; most projects move to a self-built LGPL stack (next row). |
| A proprietary app bundling a **self-built LGPL** libmpv (`-Dgpl=false`) + LGPL FFmpeg (no `--enable-gpl`/`--enable-nonfree`), dynamically linked | LGPL-2.1+ for the libs | License texts + notices; source (or written offer) for your exact libmpv/FFmpeg builds; keep `libmpv.so` a replaceable dynamic library; no anti-reverse-engineering EULA clause covering the library parts. |
| Anything containing `--enable-nonfree` FFmpeg | Non-redistributable | None available — that flag exists because the result cannot be redistributed. Rebuild without it. |
| Nothing (internal/server/personal use) | — | None. |

### Starting points for a proprietary app

Not a complete compliance program, but the topics one has to cover:

- Self-built libmpv (`-Dgpl=false`) and FFmpeg (LGPL-only flags), with
  the exact sources and flags archived — that archive backs your source
  offer.
- Dynamic linking for libmpv (the default here).
- LGPL-2.1 text, notices, and the source offer shipped with the app.
- An EULA review for reverse-engineering clauses that would conflict
  with §6.
- A decision on `--enable-gpl` codecs (x264…) — enabling them changes
  the licensing picture, not just the feature set.

## Why the whole Rust tree is permissive

mpv publishes its client API headers under ISC precisely so third-party
bindings can exist without inheriting mpv's copyleft. `rsmpv-sys` is a
clean-room FFI layer written from those headers (and nothing else);
`rsmpv` and `mpv-engine` build on it under MIT OR Apache-2.0. Any
copyleft in your final binary comes from the libmpv and FFmpeg builds
you link — never from the Rust side.
