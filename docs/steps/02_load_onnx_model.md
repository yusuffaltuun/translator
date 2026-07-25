# Step 2 — Load Local ONNX Model & Perform Translation (Micro-Step 1 of N)

## 1. Summary

Added `ort = "2.0.0-rc.12"` dependency to `Cargo.toml`. Created `src/translator.rs` with `Translator` struct, `load_model()`, and `translate()` stubs. Added `mod translator;` to `src/main.rs`.

## 2. Purpose

Establishes the ONNX Runtime integration foundation. The `Translator` struct will hold a loaded model session; `load_model()` will read an `.onnx` file from disk; `translate()` will run inference.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **ONNX Runtime (`ort`)** | Microsoft's cross-platform ML inference engine. The `ort` crate provides safe Rust bindings to run `.onnx` models locally. |
| **`ort::session::Session`** | Represents a loaded ONNX model ready for inference. Created via `Session::builder()?.commit_from_file(path)`. |
| **`SessionBuilder`** | Builder pattern for configuring and loading a model — set graph optimization level, execution providers, etc. |
| **`commit_from_file()`** | Finalizes the builder, loads the `.onnx` file from disk, and returns a `Session`. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `Cargo.toml` | Added `[dependencies]` section with `ort = "2.0.0-rc.12"` |
| **Add** | `src/translator.rs` | 20 lines — `Translator` struct, `load_model()`, `translate()` stubs |
| **Modify** | `src/main.rs` | Added `mod translator;` (line 3) |

## 5. Git Guidance

```bash
git add Cargo.toml src/translator.rs src/main.rs && git commit -m "chore: add ort crate dependency and translator module stub"
```

**Plain English:** Save the ONNX Runtime dependency and empty translator module.
