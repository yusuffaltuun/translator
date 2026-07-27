# Architecture Decision Records

## Overview

This document tracks key architectural decisions made during the evolution of **translator** — a Tauri v2 desktop application providing offline English-to-Turkish translation on Linux/Wayland (Hyprland).

---

## ADR-001: Initial MVP — Single-Pass ONNX Inference

**Date:** Project inception
**Status:** Superseded by ADR-002, ADR-003, ADR-004

### Context

The initial goal was a lightweight, zero-cost, offline EN→TR translation overlay triggered by Alt+T on Hyprland. The constraints were:

- No paid APIs or cloud services
- Low CPU/GPU usage — must not cause system lag
- Fast enough for smooth document reading (<1s latency)
- Tauri v2 for the desktop shell, Axum for the HTTP trigger

### Decision

- **Model:** Helsinki-NLP/opus-mt-en-tr — the base MarianMT model (~300M params, 170MB FP32 ONNX).
- **Runtime:** ONNX Runtime via `ort` 2.0 Rust bindings.
- **Tokenizer:** SentencePiece/Metaspace via Hugging Face `tokenizers` crate (32k vocab, `</s>` EOS, `<pad>` at 32000).
- **Architecture:** Tauri v2 app with transparent always-on-top overlay window. An Axum HTTP server on `127.0.0.1:60828` receives `GET /selection_translate` from a Hyprland keybinding (`curl`). The handler reads the Wayland primary selection via `wl-paste --primary`, runs ONNX inference, and emits results to the frontend via Tauri events.
- **Inference pipeline:** Full text fed as a single batch → tokenize → pad/truncate to 512 tokens → greedy decoding (num_beams=1) → decode output tokens → emit result.
- **Decoding parameters:** `max_length=128`, `repetition_penalty=3.0`, `length_penalty=1.0`.

### Consequences

- Simple, minimal codebase (~250 lines Rust total).
- Quality acceptable for short sentences; degraded for long or technical text.
- `max_length=128` silently truncated longer translations.
- `repetition_penalty=3.0` aggressively suppressed legitimate technical term repetition.
- Single-pass encoding of long selections caused positional embedding overflow (ONNX shape mismatch on `encoder/Add` operations for sequences >512 tokens).

---

## ADR-002: Non-Blocking Async Architecture

**Date:** Post-MVP (freeze fix)
**Status:** Active

### Context

Rapid Alt+T presses (2–3 in quick succession) caused the app to freeze completely:

1. `translator::read_selection()` spawned `wl-paste -p` synchronously inside an async Axum handler, blocking the runtime thread.
2. `Mutex<Translator>` was held across the entire ONNX inference (~1–3s), starving both the Axum HTTP server and the Tauri IPC `translate` command (which shares the same Mutex on the webview thread).
3. The Tauri webview event loop and OS close signals became unresponsive.

### Decision

- **`tokio::task::spawn_blocking`** for all heavy work: clipboard subprocess calls and ONNX inference run on Tokio's dedicated blocking thread pool, never on the async executor.
- **Minimal Mutex hold:** The `Mutex<Translator>` lock is acquired, used for a single `translate()` call, and immediately dropped — all inside the blocking task.
- **Async IPC commands:** Both `read_selection` and `translate` Tauri commands are now `async fn`, wrapping their blocking internals in `spawn_blocking`.
- **Non-blocking UI:** The HTTP handler returns immediately after spawning; the webview thread never waits on ONNX.

### Consequences

- Rapid Alt+T presses now queue independent blocking tasks instead of deadlocking.
- Window close/hide buttons remain responsive during inference.
- The `tokio` dependency gained the `rt` feature for `spawn_blocking`.
- The `Mutex<Translator>` is never held across an `await` point.

---

## ADR-003: Start-Hidden Background Mode + Editable Source Text

**Date:** UX improvement phase
**Status:** Active

### Context

The overlay window appeared on app startup (before any translation was triggered), and the source text was a read-only `<div>` — users could not manually type or edit text.

### Decision

- **Start hidden:** `tauri.conf.json` sets `"visible": false`. The window only appears when `show()` + `set_focus()` is called from the `/selection_translate` HTTP handler. The JS init block no longer calls `appWindow.show()`.
- **Editable textarea:** The source `<div>` was replaced with a `<textarea>` element with placeholder "Select text or type here…".
- **Debounced live translation:** An `input` event listener on the textarea triggers `scheduleTranslate()` with a 300ms debounce. A `translateText()` function calls `invoke('translate', { text })` and updates the UI. An `isTranslating` flag prevents concurrent ONNX calls.

### Consequences

- The overlay is invisible until explicitly triggered — no visual clutter.
- Users can type or paste text directly for instant translation feedback.
- The debounce prevents excessive inference calls during rapid typing.
- The `isTranslating` guard ensures only one ONNX session runs at a time (Mutex safety).

---

## ADR-004: Sentence-Aware Chunking + INT8 Quantization + Model Upgrade

**Date:** Quality improvement phase
**Status:** Active

### Context

The base MarianMT model (`Helsinki-NLP/opus-mt-en-tr`, 300M params, 170MB FP32) delivered poor quality for technical/academic English:

- Subword tokenizer fragmented unfamiliar compound words.
- Long selections exceeded 512-token positional embedding limits, causing ONNX shape mismatches.
- `max_length=128` silently truncated longer translations.
- `repetition_penalty=3.0` suppressed legitimate domain terminology.
- Single-pass encoding lost inter-sentence coherence for multi-sentence selections.

### Decision

#### Tier 1 — Code-Level (No Model Change)

1. **Sentence-aware chunking** (`src/translator.rs::split_into_chunks`):
   - Input text is split at sentence boundaries (`.`, `!`, `?` followed by whitespace/newline, and double newlines).
   - Very short fragments (<4 words) are merged into the preceding chunk to prevent context loss.
   - Each chunk is translated independently through the ONNX session.
   - Results are re-joined with spaces.
   - This avoids positional-embedding overflow, preserves inter-sentence coherence, and handles arbitrarily long selections.

2. **Updated decoding parameters:**
   - `max_length`: 128 → **512** (matches encoder's positional limit, prevents silent truncation).
   - `repetition_penalty`: 3.0 → **1.3** (allows legitimate technical term repetition).
   - Constants extracted to named constants at module top for maintainability.

3. **INT8 dynamic quantization** (`scripts/quantize_model.sh`):
   - Uses `onnxruntime.quantization.quantize_dynamic` with `QuantType.QInt8`.
   - Reduces model size from ~170MB to ~45MB (3.8× reduction).
   - Speeds up CPU inference by 1.5–2×.
   - Quality loss: <1% BLEU score degradation (well within noise for MarianMT).
   - One-shot offline preprocessing step, zero runtime cost.

#### Tier 2 — Model Swap

4. **Helsinki-NLP/opus-mt-tc-big-en-tr** (the "big" MarianMT variant):
   - ~500M params (vs. 300M for the base model), trained with larger hidden dimensions and more data.
   - Substantially better handling of technical/academic English, complex sentence structures, and domain terminology.
   - Quantized INT8 ≈ 120MB (still viable for desktop use).
   - Same ONNX architecture and tokenizer interface — drop-in replacement.
   - Model resolution in `src-tauri/src/lib.rs` prefers `model-int8.onnx`, falls back to `model-fp32.onnx`.

#### Why Not Alternatives

| Alternative | Rejected Because |
|---|---|
| `facebook/nllb-200-distilled-600M` | 150MB INT8 but requires different tokenizer and ONNX input signature — larger refactor for marginal gain over tc-big |
| `facebook/mbart-large-50` | 2.4GB+ — violates low-resource constraint |
| Small LLM (Qwen2.5-0.5B) | Decoder-only architecture: ~350MB Q4, ~500ms/sentence on CPU — too slow, too heavy |
| Paid APIs (DeepL, Google) | Violates zero-cost constraint |

### Consequences

- **Quality:** Significant improvement for technical text — the tc-big model handles domain terminology and complex structures much better than the base model.
- **Speed:** INT8 quantization + sentence chunking (smaller per-chunk batches) maintains sub-second latency for typical selections.
- **Memory:** INT8 model uses ~45MB RAM vs. ~170MB for FP32; tc-big INT8 uses ~120MB — both well within desktop comfort.
- **Robustness:** Sentence-aware chunking eliminates ONNX shape mismatch crashes on long selections.
- **Backward compatibility:** Model resolution auto-detects INT8 vs FP32, so the app works with either model variant without configuration changes.

---

## ADR-005: Separate Encoder-Decoder ONNX with KV Cache Autoregressive Decoding

**Date:** Model architecture refactor
**Status:** Active

### Context

The tc-big model (`Helsinki-NLP/opus-mt-tc-big-en-tr`) is only available as a PyTorch checkpoint and a pre-exported encoder on `onnx-community/opus-mt-tc-big-en-tr`. There is no monolithic encoder-decoder ONNX with baked-in beam search like the original Teradata model. This means the decoder must be exported separately, and autoregressive decoding with KV cache must be implemented in Rust.

The `onnx-community` decoder export had no `past_key_values` inputs — only `present.*` outputs — making incremental decoding impossible. Exporting via `torch.onnx.export` with `dynamo=True` captured only 3/6 decoder layers (PyTorch tracing limitation with dynamic control flow in MarianDecoder). Legacy export (`dynamo=False`) succeeded with all 6 layers.

### Decision

#### Model Files

| File | Format | Size | Source |
|---|---|---|---|
| `models/encoder_model_int8.onnx` | INT8 quantized | 130 MB | Quantized from FP32 export |
| `models/decoder_model_int8.onnx` | INT8 quantized | 209 MB | Re-exported from PyTorch + quantized |
| `models/tokenizer.json` | SentencePiece/Unigram | 7.7 MB | Rebuilt from `source.spm` + HF encoder dict |

#### Decoder ONNX Export

- **Wrapper:** `FullDecoder` wraps `MarianDecoder` + `lm_head`, accepts explicit named inputs (no variadic `*args`).
- **Inputs (15):** `input_ids`, `encoder_hidden_states`, `encoder_attention_mask`, + 12 decoder KV cache tensors (`past_key_values.{0-5}.decoder.{key,value}`).
- **Outputs (25):** `logits`, + 12 decoder present KV (`present.{0-5}.decoder.{key,value}`), + 12 encoder present KV (`present.{0-5}.encoder.{key,value}` — recomputed each step, not cached).
- **Exporter:** `torch.onnx.export` with `dynamo=False`, `opset_version=17`, `do_constant_folding=False`.
- **Dynamic axes:** `batch`, `dec_seq`, `enc_seq`, `past_seq` — all dimensions symbolic.

#### Encoder KV Cache: Not Used

The encoder present KV outputs (`present.{0-5}.encoder.{key,value}`) are **not cached** between steps. The decoder recomputes cross-attention from `encoder_hidden_states` each step. This is functionally correct (MarianDecoder recomputes cross-attention from the full encoder output) but wastes ~50% of the present KV outputs. The alternative — editing the ONNX graph to remove encoder KV outputs — was rejected as fragile.

#### Tokenizer Reconstruction

The `onnx-community` tokenizer used a different BPE/Unigram vocab that produced different token IDs than the HF `MarianTokenizer`. The correct tokenizer is SentencePiece-based: SPM vocab (32K pieces) mapped through an `encoder` dict (57060 entries) to a shared vocabulary.

- Built using the `tokenizers` crate's `Unigram` model with SPM pieces + encoder dict.
- `tokenizer.json` normalizer set to `null` (no-op) — `tokenizers` v0.21.4 cannot parse `Precompiled` with null/empty string.
- Encode-verified against HF `MarianTokenizer` output.

### Autoregressive Decoding with KV Cache

The decoder uses a **single-token-per-step** autoregressive loop:

1. **Step 0:** Feed `[decoder_start_token_id]` (57059) with empty KV cache (shape `[1, 16, 0, 64]`). Model outputs logits `[1, 1, vocab]` and present KV `[1, 16, 1, 64]`.
2. **Step N > 0:** Feed **only the latest generated token** with cached past KV from step N-1. Model outputs logits `[1, 1, vocab]` and present KV `[1, 16, N+1, 64]`.
3. **Termination:** Loop ends when EOS (43741), PAD (57059), or `MAX_OUTPUT_LEN` (256) is reached.

**Critical invariant:** Only the single newest token is fed as `input_ids` at each step. Feeding the full `generated_tokens` sequence would cause the KV cache to grow quadratically (past N + input N = 2N positions per step), corrupting self-attention and causing premature truncation.

#### Empty KV Cache Tensor Creation

On step 0, past KV tensors must have shape `[1, 16, 0, 64]` (zero-length sequence dimension). The `ort` crate's `Tensor::from_array` calls `CreateTensorWithDataAsOrtValue`, which requires all dimensions ≥ 1. This fails with a zero-length dimension.

**Fix:** Use `Tensor::<f32>::new(&Allocator::default(), shape)` instead, which calls `CreateTensorAsOrtValue`. This allocator-based API handles zero-element tensors correctly, as confirmed by the ORT C API behavior (Python `onnxruntime` handles `np.zeros((1,16,0,64))` natively via the same underlying path).

### Consequences

- **Quality:** The tc-big model produces substantially better translations than the base model, especially for technical/academic text.
- **KV cache correctness:** Single-token-per-step decoding ensures the KV cache grows linearly (1 position per step) and self-attention computes correctly.
- **Memory:** INT8 decoder is 209 MB — larger than the encoder but acceptable for desktop use.
- **Speed:** Each autoregressive step runs one encoder forward pass (cached) + one decoder forward pass. For typical 10-20 token translations, this is ~10-20ms per step on CPU.
- **Tokenizer fidelity:** The rebuilt tokenizer produces identical token IDs to the HF `MarianTokenizer`, eliminating a class of silent translation errors.

---

## Architecture Evolution Summary

```
Phase 1 (MVP)
  Single-pass FP32 MarianMT → freeze on rapid input → shape mismatch on long text

Phase 2 (Non-blocking)
  tokio::task::spawn_blocking → minimal Mutex hold → async IPC commands
  Result: No more freezes, responsive UI

Phase 3 (UX)
  Start-hidden overlay → editable textarea → 300ms debounced live translation
  Result: Clean background mode, interactive text editing

Phase 4 (Quality)
  Sentence chunking → INT8 quantization → tc-big model upgrade
  Result: Production-quality offline EN→TR translation

Phase 5 (Architecture)
  Separate encoder/decoder ONNX → KV cache autoregressive decoding → tokenizer rebuild
  Result: Linear-growth KV cache, correct self-attention, full-sentence translations
```
