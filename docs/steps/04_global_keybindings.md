# Step 4 —  (Micro-Step 1 of N)

## 1. Summary

Added `tauri-plugin-global-shortcut = "2"` dependency. Registered `Alt+T` as a system-wide hotkey via `app.global_shortcut().on_shortcut()`. On press: toggles window visibility, reads primary selection, translates via ONNX, and emits a `translate-result` event to the frontend. Frontend updated with `listen('translate-result', ...)` to receive shortcut-triggered results. Added `global-shortcut:default` and `core:window:allow-set-focus` capabilities.

## 2. Purpose

Enables the core UX: **select text anywhere → press `Alt+T` → see translation instantly**. No mouse interaction required. The shortcut works across all applications on the desktop.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **Global shortcut** | Key combination intercepted by the OS/compositor, active even when the app is not focused. |
| **`tauri-plugin-global-shortcut`** | Official Tauri v2 plugin. On Linux: X11 uses `XGrabKey`; Wayland uses the `org.freedesktop.portal.GlobalShortcuts` D-Bus portal. |
| **`on_shortcut()`** | Registers a specific key binding with a per-shortcut callback. |
| **`AtomicBool` + `fetch_xor`** | Lock-free atomic toggle — safely flips `true↔false` from any thread without a Mutex. |
| **`Emitter` trait** | Tauri v2 trait providing `app.emit()` to push events from Rust to the frontend JS. |
| **`listen()`** | Frontend JS — subscribes to events emitted from the Rust backend via `app.emit()`. |
| **`ShortcutState::Pressed`** | Filter: only act on key-down, ignore key-up releases. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src-tauri/Cargo.toml` | Added `tauri-plugin-global-shortcut = "2"` |
| **Rewrite** | `src-tauri/src/lib.rs` | 70 → 131 lines — plugin registration, `AtomicBool` toggle, shortcut handler with thread spawn, `emit()` for results |
| **Modify** | `src-tauri/capabilities/default.json` | Added `global-shortcut:default`, `core:window:allow-set-focus` |
| **Modify** | `dist/index.html` | Added `listen('translate-result')`, `updateUI()` helper, resolved `window.__TAURI__.event.listen` |

## 5. Git Guidance

```bash
git add src-tauri/Cargo.toml src-tauri/src/lib.rs src-tauri/capabilities/default.json dist/index.html docs/steps/04_global_keybindings.md && git commit -m "feat: add Alt+T global shortcut to toggle overlay and translate"
```

**Plain English:** Save the global hotkey integration and event-driven UI update.

---

# Micro-Step 2 — HTTP Trigger Server (Wayland-Compatible Alternative)

## 1. Summary

Disabled `tauri-plugin-global-shortcut` registration (dependency kept but unused). Added `axum = "0.8"` and `tokio = "1"` dependencies. Created a local HTTP server on `127.0.0.1:60828` with `GET /selection_translate` endpoint. The endpoint toggles overlay visibility, reads primary selection via `wl-paste -p`, runs ONNX translation, and emits results to the frontend via `app.emit()`. Shared a single `Arc<Mutex<Translator>>` between Tauri IPC commands and the HTTP handler to avoid loading the 170MB model twice. Removed `global-shortcut:default` from capabilities.

## 2. Purpose

`tauri-plugin-global-shortcut` relies on the XDG GlobalShortcuts portal which is unreliable on Hyprland/Wayland. The HTTP server approach is compositor-agnostic — any tool (Hyprland keybinding, shell script, `curl`) can trigger translation via a simple HTTP GET. This is the Wayland-compatible path.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **Axum** | Tokio-based HTTP framework by the Tokio team. Lightweight, async, production-grade. |
| **`GET /selection_translate`** | HTTP endpoint — triggered by `curl http://127.0.0.1:60828/selection_translate` from any keybinding tool. |
| **`Arc<Mutex<T>>`** | Shared ownership (`Arc`) + mutual exclusion (`Mutex`). One `Translator` instance shared between Tauri IPC and the HTTP server. |
| **`AtomicBool::fetch_xor`** | Lock-free toggle — flips `false→true→false→…` atomically. Used for overlay visibility state. |
| **`tauri::async_runtime::spawn`** | Runs an async task on Tauri's internal Tokio runtime — no need to create a separate runtime. |
| **Compositor-agnostic** | Works on any Wayland compositor (Hyprland, Sway, GNOME, KDE) without portal dependencies. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src-tauri/Cargo.toml` | Added `axum = "0.8"`, `tokio = { version = "1", features = ["net"] }` |
| **Rewrite** | `src-tauri/src/lib.rs` | 131 → 155 lines — removed shortcut plugin, added `AppState`, `selection_translate()` handler, HTTP server on `:60828`, shared `Arc<Mutex<Translator>>` |
| **Modify** | `src-tauri/capabilities/default.json` | Removed `global-shortcut:default` |

## 5. How to Trigger

```bash
# Hyprland keybinding (add to ~/.config/hyprland/hyprland.conf):
bind = ALT, T, exec, curl -s http://127.0.0.1:60828/selection_translate > /dev/null

# Or from any shell:
curl http://127.0.0.1:60828/selection_translate
```

## 6. Git Guidance

```bash
git add src-tauri/Cargo.toml src-tauri/src/lib.rs src-tauri/capabilities/default.json docs/steps/04_global_keybindings.md && git commit -m "feat: replace global-shortcut with http trigger server for wayland"
```

**Plain English:** Save the HTTP server approach that works on all Wayland compositors.
