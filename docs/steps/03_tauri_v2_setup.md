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

---

# Micro-Step 2 — Wire Existing Rust Logic into Tauri as Commands

## 1. Summary

Converted the root crate into a library by adding `src/lib.rs` (extracting all selection + translation logic) and replacing `src/main.rs` with a thin binary. Added `translator = { path = ".." }` to `src-tauri/Cargo.toml`. Created two Tauri commands (`read_selection`, `translate`) in `src-tauri/src/lib.rs` with managed `TranslatorState` via `Mutex`. Updated `capabilities/default.json` with window permissions. Both root crate and `src-tauri` crate compile successfully.

## 2. Purpose

The frontend needs to call Rust functions (selection reading, translation) via Tauri's IPC system. By converting the root crate to a library, both the CLI binary and Tauri share identical code without duplication. The `#[tauri::command]` attribute exposes Rust functions as async-callable IPC endpoints from the webview.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **Path dependency** | `translator = { path = ".." }` — Cargo dependency pointing to a local crate on disk rather than crates.io. |
| **`crate-type = ["lib", "cdylib", "staticlib"]`** | Compiles as a Rust lib, C dynamic lib, and C static lib — needed for both Cargo consumers and FFI/mobile targets. |
| **`#[tauri::command]`** | Attribute macro that generates IPC serialization/deserialization boilerplate for a Rust function, making it callable from JS. |
| **`State<'_, T>`** | Tauri's managed state injector. The framework owns the `T` and provides borrows to commands via dependency injection. |
| **`Mutex<T>`** | Mutual exclusion lock. Required because `Translator::translate()` takes `&mut self` — Tauri commands run on a thread pool. |
| **`Manager` trait** | Provides `app.manage()`, `app.path()`, etc. Must be imported explicitly in Tauri v2. |
| **`app.manage()`** | Registers a state object with Tauri's DI container. Later retrieved in commands via `State<'_, T>`. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Add** | `src/lib.rs` | 83 lines — public `mod translator`, `DisplayServer` enum, `detect_display_server()`, `read_selection()`, `main()` |
| **Rewrite** | `src/main.rs` | 94 → 3 lines — thin binary calling `translator::main()` |
| **Modify** | `Cargo.toml` | Added `[lib]` section with `crate-type = ["lib", "cdylib", "staticlib"]` |
| **Modify** | `src-tauri/Cargo.toml` | Added `translator = { path = ".." }` dependency |
| **Rewrite** | `src-tauri/src/lib.rs` | 16 → 65 lines — `TranslatorState`, `read_selection` command, `translate` command, model loading in `setup()` |
| **Modify** | `src-tauri/capabilities/default.json` | 11 → 18 lines — added window manipulation permissions |

## 5. Git Guidance

```bash
git add src/lib.rs src/main.rs Cargo.toml src-tauri/Cargo.toml src-tauri/src/lib.rs src-tauri/capabilities/default.json && git commit -m "feat: wire selection and translation logic into tauri commands"
```

**Plain English:** Save the library conversion, Tauri commands, and capability permissions.

---

# Micro-Step 3 — Floating Overlay UI Frontend

## 1. Summary

Updated `src-tauri/tauri.conf.json` to configure a frameless, transparent, always-on-top window (380x220, hidden on start). Rewrote `dist/index.html` with a functional overlay UI: dark theme card with drag handle, source/translation text areas, "Translate Selection" button, and close button. Frontend calls `invoke('read_selection')` and `invoke('translate', { text })` via Tauri v2's injected IPC bridge.

## 2. Purpose

This is the visual layer of the floating overlay. The user selects text anywhere on screen, then clicks "Translate Selection" — the frontend calls the Rust backend, receives the translation, and displays it in a non-intrusive floating card. The window starts hidden (`visible: false`) and is shown via `appWindow.show()` after DOM loads — Step 4 will tie this to a global hotkey.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **Frameless window** | `"decorations": false` — removes title bar and OS window chrome. The app draws its own header. |
| **Transparent window** | `"transparent": true` — the webview background is alpha-transparent. Only the drawn content is visible. |
| **Always-on-top** | `"alwaysOnTop": true` — window stays above all other windows. |
| **`skipTaskbar`** | Hides the window from the taskbar/dock — appropriate for overlay/utility windows. |
| **`visible: false`** | Window created hidden. Shown programmatically via `appWindow.show()`. |
| **`window.__TAURI__`** | Tauri v2's injected IPC bridge. Provides `core.invoke()` for calling Rust commands and `window` module for window control. |
| **`startDragging()`** | Tauri API — begins window drag from the next mouse event. Used for custom title bar drag regions. |
| **`getCurrentWindow()`** | Returns the `WebviewWindow` handle for the current webview, enabling `.show()`, `.hide()`, `.startDragging()`. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src-tauri/tauri.conf.json` | 37 → 40 lines — transparent, frameless, always-on-top, skipTaskbar, visible:false, identifier update |
| **Rewrite** | `dist/index.html` | 26 → 148 lines — full overlay UI with drag, translate, close, status feedback |

## 5. Git Guidance

```bash
git add src-tauri/tauri.conf.json dist/index.html docs/steps/03_tauri_v2_setup.md && git commit -m "feat: implement floating overlay ui frontend"
```

**Plain English:** Save the overlay window config and the interactive HTML frontend.
