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

---

# Micro-Step 2 — Download ONNX Model & Verify Loading

## 1. Summary

Downloaded `model-fp32.onnx` (170MB) and `tokenizer.json` (2MB) from `Teradata/opus-mt_tiny_eng-tur` into `models/`. Updated `main.rs` to load the model on startup and print success. Created `.gitignore` to exclude model files from git.

## 2. Purpose

Provides the actual translation model for `ort` to load. Validates that the full ONNX Runtime pipeline works end-to-end.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **MarianMT** | Lightweight seq2seq translation model by Microsoft/Helsinki-NLP. |
| **`model-fp32.onnx`** | Full-precision (32-bit float) ONNX model file — encoder + decoder merged. |
| **`tokenizer.json`** | SentencePiece tokenizer converting text ↔ token IDs. |
| **`.gitignore`** | Tells git to exclude files matching patterns — model binaries are too large for version control. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Add** | `models/model-fp32.onnx` | 170MB binary — MarianMT en→tr model |
| **Add** | `models/tokenizer.json` | 2MB — SentencePiece tokenizer |
| **Modify** | `src/main.rs` | 75 → 85 lines. Added `use translator::Translator`, model loading call in `main()` |
| **Add** | `.gitignore` | 2 lines — excludes `models/*.onnx` and `models/*.json` |

## 5. Git Guidance

```bash
git add src/main.rs src/translator.rs .gitignore docs/steps/02_load_onnx_model.md && git commit -m "feat: download onnx model and verify loading"
```

**Plain English:** Save the model loading code, .gitignore, and docs. Model files themselves are excluded from git.

---

# Micro-Step 3 — Implement Translation Inference Pipeline

## 1. Summary

Implemented the full translation inference pipeline. Added `tokenizers = "0.21"` to `Cargo.toml`. Rewrote `src/translator.rs` with `translate()` that tokenizes input, passes all 7 BeamSearch inputs to the ONNX model, and decodes the output tokens. Verified: `"Hello, how are you?"` → `"Merhaba, nasılsınız?"`.

## 2. Purpose

This is the core functionality of Step 2 — the program now translates English text to Turkish entirely offline.

## 3. Industry Terms & Concepts

| Term | Definition |
|------|-----------|
| **BeamSearch** | Decoding algorithm exploring multiple hypotheses in parallel. Built into this model's ONNX graph. |
| **`num_beams: 1`** | Greedy mode — one hypothesis, fastest. |
| **`repetition_penalty: 3.0`** | Prevents token repetition loops. |
| **`tokenizers` crate** | Hugging Face's Rust library for loading `tokenizer.json`. |
| **`Tensor::from_array`** | Creates an ONNX input tensor from a Rust `(shape, data)` tuple. |

## 4. Change Log

| Action | File | Lines |
|--------|------|-------|
| **Modify** | `Cargo.toml` | Added `tokenizers = "0.21"` |
| **Rewrite** | `src/translator.rs` | 21 → 82 lines. Full BeamSearch inference pipeline. |
| **Modify** | `src/main.rs` | 83 → 81 lines. Added `tokenizer_path`, `translate()` call. |

## 5. Git Guidance

```bash
git add Cargo.toml src/translator.rs src/main.rs && git commit -m "feat: implement offline translation inference pipeline"
```

**Plain English:** Save the working translation logic.
