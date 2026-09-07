//! cfg aliases for the export backend's feature+platform gate.
//!
//! The exported-frame backend only exists when the `export` feature is on
//! *and* the target is one of the three platforms that implement it. That
//! predicate — `all(feature = "export", any(target_os = "macos",
//! target_os = "linux", target_os = "windows"))` — otherwise has to be
//! retyped at every gated item; a single `#[cfg(export_backend)]` is both
//! readable and impossible to get subtly wrong. `wgpu_backend` is the same
//! for the `wgpu` feature (which implies `export`).
//!
//! No dependencies: this uses only the environment Cargo already exports
//! (`CARGO_FEATURE_*`, `CARGO_CFG_TARGET_OS`), so the crate's "boring
//! default build" policy is untouched — a build script that pulls in no
//! crates adds nothing to any consumer's tree.

fn main() {
    // Register the custom cfgs so `unexpected_cfgs` stays quiet under the
    // `-D warnings` CI lint (Rust 1.80+).
    println!("cargo::rustc-check-cfg=cfg(export_backend)");
    println!("cargo::rustc-check-cfg=cfg(wgpu_backend)");

    let supported = matches!(
        std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("macos" | "linux" | "windows")
    );
    // CARGO_FEATURE_<NAME> is present exactly when that feature is enabled.
    let export = std::env::var_os("CARGO_FEATURE_EXPORT").is_some();
    let wgpu = std::env::var_os("CARGO_FEATURE_WGPU").is_some();

    if supported && export {
        println!("cargo::rustc-cfg=export_backend");
    }
    if supported && wgpu {
        println!("cargo::rustc-cfg=wgpu_backend");
    }

    println!("cargo::rerun-if-changed=build.rs");
}
