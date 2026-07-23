# Step 1 — Read Linux Primary Selection (Micro-Step 1 of N)

## 1. Summary

Created the Rust project scaffold: `Cargo.toml` and `src/main.rs` with a minimal `main()` function. Verified it compiles successfully.

## 2. Purpose

Establishes the project foundation so all subsequent selection-reading code can be added incrementally into `src/main.rs`.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **`cargo build`** | Compiles the Rust crate. Confirms `Cargo.toml` and source files are valid. |
| **`edition = "2021"`** | The Rust language edition — controls syntax/API defaults. 2021 is the current stable edition. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Add** | `Cargo.toml` | 4 lines (package name, version, edition) |
| **Add** | `src/main.rs` | 3 lines (`fn main()` + placeholder print) |

## 5. Git Guidance

```bash
git add Cargo.toml src/main.rs
git commit -m "chore: initialize rust project scaffold"
```

**Plain English:** Stage and commit the two new files as a clean save-point.

---

# Micro-Step 2 — Detect Display Server (X11 / Wayland)

## 1. Summary

Added display-server detection to `src/main.rs`. The program reads `XDG_SESSION_TYPE`, `WAYLAND_DISPLAY`, and `DISPLAY` environment variables to determine if the session is Wayland, X11, or Unknown.

## 2. Purpose

Selection capture uses different protocols depending on the display server. Knowing which one is active routes to the correct reader in later micro-steps.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **`XDG_SESSION_TYPE`** | Env var set by Linux display managers — `"x11"` or `"wayland"`. |
| **`WAYLAND_DISPLAY`** | Env var set when a Wayland compositor is running. |
| **`DISPLAY`** | Env var set when an X11 server is running. |
| **`enum`** | Rust type that can be one of several variants. |
| **`match`** | Rust pattern-matching control flow. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src/main.rs` | 3 → 33 lines. Added `use std::env`, `enum DisplayServer`, `detect_display_server()`, updated `main()`. |

## 5. Git Guidance

```bash
git add src/main.rs && git commit -m "feat: detect X11/Wayland display server"
```

**Plain English:** Save the display-server detection logic.

---

# Micro-Step 3 — Read X11 Primary Selection via xclip

## 1. Summary

Added `read_x11_selection()` function to `src/main.rs` that spawns `xclip -selection primary -o` and captures its stdout. Updated `main()` to call it when X11 is detected.

## 2. Purpose

Provides the X11 path for reading the primary selection. One of two display-server branches; Wayland follows next.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **`Command::new("xclip")`** | Rust's `std::process::Command` — spawns a child process. |
| **`.args([...])`** | Passes CLI arguments to the spawned process. |
| **`.output()`** | Runs the process to completion and captures stdout/stderr. |
| **`Result<String, String>`** | Rust's `Result` type — `Ok(String)` on success, `Err(String)` on failure. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src/main.rs` | 29 → 51 lines. Added `use std::process::Command`, `read_x11_selection()`, updated `main()` with match arms. |

## 5. Git Guidance

```bash
git add src/main.rs && git commit -m "feat: read X11 primary selection via xclip"
```

**Plain English:** Save the X11 clipboard-reading logic.

---

# Micro-Step 4 — Read Wayland Primary Selection via wl-paste

## 1. Summary

Added `read_wayland_selection()` function to `src/main.rs` that spawns `wl-paste --primary` and captures stdout. Updated the `Wayland` match arm in `main()` to call it.

## 2. Purpose

Completes the Wayland path — the system's actual display server. The program now reads primary selection on both X11 and Wayland.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **`wl-paste --primary`** | Reads the primary selection buffer on Wayland via the `wlr-data-control` protocol. |
| **`wl-clipboard`** | Package providing `wl-copy` and `wl-paste` — standard Wayland clipboard CLI tools. |
| **`map_err`** | Converts one error type to another — maps `io::Error` to a human-readable `String`. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `src/main.rs` | 58 → 71 lines. Added `read_wayland_selection()`, updated `Wayland` match arm. |

## 5. Git Guidance

```bash
git add src/main.rs && git commit -m "feat: read Wayland primary selection via wl-paste"
```

**Plain English:** Save the Wayland clipboard-reading logic.
