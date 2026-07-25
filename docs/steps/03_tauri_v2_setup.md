# Step 3 — Setup Tauri v2 for Floating Overlay UI (Micro-Step 1 of N)

## 1. Summary

Installed `@tauri-apps/cli` v2.11.4 globally via npm. Ran `npx tauri init --ci` to scaffold `src-tauri/` containing `tauri.conf.json`, `Cargo.toml` (with `tauri = "2.11.3"`), `src/lib.rs`, `src/main.rs`, `build.rs`, and `capabilities/default.json`. Created `dist/index.html` as the minimal frontend placeholder. Verified compilation with `cargo build` in `src-tauri/` — all 425 crates compiled successfully.

## 2. Purpose

Tauri v2 provides the desktop app shell for the floating overlay UI. It embeds a WebKitGTK webview on Linux to render HTML/CSS/JS, while allowing direct Rust function calls from the frontend via Tauri commands. The existing `src/` translation code is preserved untouched — it will be wired into the Tauri app builder in the next micro-step.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **Tauri v2** | Rust framework for desktop apps using a native webview. v2 adds mobile support, capability-based permissions, and a plugin system. |
| **`src-tauri/`** | Tauri's Rust backend directory. Separate `Cargo.toml` with its own dependency tree, isolated from the root project. |
| **`tauri.conf.json`** | Central config: window dimensions, security CSP, build paths, bundle metadata, app identifier. |
| **`lib.rs`** | App entrypoint — `tauri::Builder::default().run(generate_context!())`. `#[cfg_attr(mobile, mobile_entry_point)]` enables mobile builds. |
| **`main.rs`** | Thin wrapper calling `app_lib::run()` — satisfies both desktop binary and mobile library targets. |
| **`capabilities/default.json`** | Tauri v2 permission system — declares which IPC permissions the frontend has (here: `core:default`). |
| **`frontendDist`** | Path to static web assets served by the webview, relative to `src-tauri/`. Set to `../dist`. |
| **WebKitGTK** | Linux webview engine used by Tauri/WRY. Renders HTML/CSS/JS natively without bundling Chromium. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Add** | `src-tauri/tauri.conf.json` | 37 lines — window config, security, build paths, bundle metadata |
| **Add** | `src-tauri/Cargo.toml` | 25 lines — `tauri = "2.11.3"`, `tauri-build = "2.6.3"`, `tauri-plugin-log`, `serde`, `serde_json` |
| **Add** | `src-tauri/src/lib.rs` | 16 lines — Tauri builder with log plugin in debug mode |
| **Add** | `src-tauri/src/main.rs` | 6 lines — thin `main()` calling `app_lib::run()` |
| **Add** | `src-tauri/build.rs` | 3 lines — `tauri_build::build()` |
| **Add** | `src-tauri/capabilities/default.json` | 11 lines — `core:default` permissions for `main` window |
| **Add** | `src-tauri/.gitignore` | Tauri-generated ignores |
| **Add** | `src-tauri/icons/` | Default Tauri app icons |
| **Add** | `dist/index.html` | 30 lines — minimal frontend placeholder with transparent background |

## 5. Git Guidance

```bash
git add src-tauri/ dist/ && git commit -m "chore: initialize tauri v2 scaffolding"
```

**Plain English:** Save the Tauri v2 project structure, icons, config, and placeholder frontend.
