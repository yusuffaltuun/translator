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
