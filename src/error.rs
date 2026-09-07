use thiserror::Error;

/// Everything this crate's fallible calls can return.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An error from libmpv itself, via rsmpv (its `Display` carries
    /// mpv's `mpv_error_string` text plus the numeric code).
    #[error("mpv: {0}")]
    Mpv(#[from] rsmpv::Error),
    /// An attach called while a render context is already live. mpv
    /// supports exactly one render context per handle, of either backend.
    #[error("a render context is already attached")]
    AlreadyAttached,
    /// A render call against the wrong attached backend (e.g. `render_gl`
    /// while the software context is attached) — a wiring bug in the
    /// shell, surfaced loudly rather than silently dropped.
    #[error("render call does not match the attached render backend")]
    RenderBackendMismatch,
    /// Setting up the exported-frame backend's hidden GL context or its
    /// exportable framebuffers (IOSurface / DMA-BUF / shared D3D11
    /// texture) failed (`export` feature,
    /// [`Engine::attach_exported_render`](crate::Engine::attach_exported_render)).
    /// The common non-bug cause is a session without GPU access —
    /// no WindowServer on macOS, no readable DRM render node on Linux
    /// (SSH, bare CI, missing render/video group), no OpenGL ICD or no
    /// `WGL_NV_DX_interop2` on Windows — treat it like a missing display
    /// and fall back to another backend or skip.
    #[cfg(export_backend)]
    #[error("exported render setup: {0}")]
    ExportSetup(String),
    /// Importing an exported frame into a wgpu device failed (`wgpu`
    /// feature,
    /// [`ExportedFrame::into_wgpu_texture`](crate::ExportedFrame::into_wgpu_texture))
    /// — the device isn't on the platform's native backend (Metal on
    /// macOS, Vulkan on Linux, DX12 on Windows), lacks a required wgpu
    /// feature (`VULKAN_EXTERNAL_MEMORY_DMA_BUF` on Linux), sits on a
    /// different GPU than the engine's hidden context (Windows), or the
    /// driver refused the IOSurface wrap / DMA-BUF import / shared-handle
    /// open.
    #[cfg(wgpu_backend)]
    #[error("wgpu import: {0}")]
    WgpuImport(String),
    /// A call that needs a live render context ran before any attach.
    /// Today only
    /// [`Engine::set_render_update_callback`](crate::Engine::set_render_update_callback)
    /// returns this — a callback that could never fire is a wiring bug,
    /// surfaced loudly like
    /// [`RenderBackendMismatch`](Self::RenderBackendMismatch). The frame
    /// path deliberately does *not* use it: unattached,
    /// `render_gl`/`render_sw` return `Ok` untouched and `render_update`
    /// returns `false`, because "not attached yet" is an ordinary
    /// startup state there, not a bug — don't match on this variant to
    /// detect a missing attach from a draw handler.
    #[error("no render context is attached")]
    NotAttached,
}

/// Shorthand for results carrying this crate's [`enum@Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Diagnostic text for a raw `client.h` `mpv_error` code: mpv's own
/// `mpv_error_string` text plus the numeric code (rsmpv's `Display`
/// carries both) — not user-facing copy.
pub(crate) fn describe_code(code: i32) -> String {
    rsmpv::Error::from_raw(code).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_code_carries_mpv_text_and_code() {
        // -17 is MPV_ERROR_UNKNOWN_FORMAT; the exact wording belongs to
        // mpv, so pin only that its text came through alongside the code.
        let msg = describe_code(-17);
        assert!(msg.contains("format"), "unexpected message: {msg}");
        assert!(msg.contains("-17"), "unexpected message: {msg}");
        // Unknown codes still produce a code-bearing string.
        assert!(describe_code(-99).contains("-99"));
    }
}
