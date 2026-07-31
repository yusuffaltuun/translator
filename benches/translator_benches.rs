use criterion::{black_box, criterion_group, criterion_main, Criterion};

use translator::traits::{InferenceEngine, Tokenizer};
use translator::translator::mocks::{MockInferenceEngine, MockTokenizer};
use translator::translator::{HuggingFaceTokenizer, OnnxInferenceEngine, Translator};

// ──────────────────────────────────────────────
//  Model path detection (for Layer 3 & 4)
// ──────────────────────────────────────────────

fn real_paths() -> Option<(&'static str, &'static str, &'static str)> {
    let candidates = [
        (
            "models/encoder_model_int8.onnx",
            "models/decoder_model_int8.onnx",
            "models/tokenizer.json",
        ),
        (
            "models/encoder_model.onnx",
            "models/decoder_model.onnx",
            "models/tokenizer.json",
        ),
    ];
    for &(enc, dec, tok) in &candidates {
        if std::path::Path::new(enc).exists()
            && std::path::Path::new(dec).exists()
            && std::path::Path::new(tok).exists()
        {
            return Some((enc, dec, tok));
        }
    }
    None
}

// ══════════════════════════════════════════════
//  Layer 1 – Tokenizer & Text Handling
// ══════════════════════════════════════════════

fn bench_split_into_chunks(c: &mut Criterion) {
    let text = "This is a long text. It has multiple sentences! And questions? Yes, it does.\n\nNew paragraph here. With more sentences to split. Let's see how fast this goes.";
    c.bench_function("split_into_chunks", |b| {
        b.iter(|| {
            let chunks = Translator::<MockTokenizer, MockInferenceEngine>::split_into_chunks(
                black_box(text),
            );
            black_box(chunks)
        })
    });
}

fn bench_mock_encode(c: &mut Criterion) {
    let tokenizer = MockTokenizer::new();
    c.bench_function("mock_encode", |b| {
        b.iter(|| {
            let ids = tokenizer.encode(black_box("hello world")).unwrap();
            black_box(ids)
        })
    });
}

fn bench_mock_decode(c: &mut Criterion) {
    let tokenizer = MockTokenizer::new();
    let ids = vec![1, 2, 3];
    c.bench_function("mock_decode", |b| {
        b.iter(|| {
            let text = tokenizer.decode(black_box(&ids)).unwrap();
            black_box(text)
        })
    });
}

fn bench_real_encode_decode(c: &mut Criterion) {
    let (_, _, tok) = match real_paths() {
        Some(p) => p,
        None => return,
    };
    let tokenizer = HuggingFaceTokenizer::from_file(tok).unwrap();

    c.bench_function("real_encode", |b| {
        b.iter(|| {
            let ids = tokenizer
                .encode(black_box("Hello, how are you? I am fine, thank you."))
                .unwrap();
            black_box(ids)
        })
    });

    let ids = tokenizer
        .encode("Hello, how are you? I am fine, thank you.")
        .unwrap();
    c.bench_function("real_decode", |b| {
        b.iter(|| {
            let text = tokenizer.decode(black_box(&ids)).unwrap();
            black_box(text)
        })
    });
}

// ══════════════════════════════════════════════
//  Layer 2 – Mock Pipeline (Rust orchestration)
// ══════════════════════════════════════════════

fn bench_mock_translate(c: &mut Criterion) {
    let tokenizer = MockTokenizer::new();
    let engine = MockInferenceEngine::new();
    let mut translator = Translator::new(tokenizer, engine);
    c.bench_function("mock_translate", |b| {
        b.iter(|| {
            let result = translator
                .translate(black_box("Hello, how are you?"))
                .unwrap();
            black_box(result)
        })
    });
}

// ══════════════════════════════════════════════
//  Layer 3 – Pure ONNX Inference
// ══════════════════════════════════════════════

fn bench_onnx_run_inference(c: &mut Criterion) {
    let (enc, dec, tok) = match real_paths() {
        Some(p) => p,
        None => return,
    };
    let mut engine = OnnxInferenceEngine::load(enc, dec).unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(tok).unwrap();
    let input_ids = tokenizer.encode("Hello, how are you?").unwrap();

    c.bench_function("onnx_run_inference", |b| {
        b.iter(|| {
            let ids = engine.run_inference(black_box(&input_ids)).unwrap();
            black_box(ids)
        })
    });
}

// ══════════════════════════════════════════════
//  Layer 4 – End-to-End Real Translation
// ══════════════════════════════════════════════

fn bench_end_to_end_translate(c: &mut Criterion) {
    let (enc, dec, tok) = match real_paths() {
        Some(p) => p,
        None => return,
    };
    let mut translator =
        Translator::<HuggingFaceTokenizer, OnnxInferenceEngine>::load_model(enc, dec, tok).unwrap();

    c.bench_function("end_to_end_translate", |b| {
        b.iter(|| {
            let result = translator
                .translate(black_box("Hello, how are you?"))
                .unwrap();
            black_box(result)
        })
    });
}

// ──────────────────────────────────────────────
//  Criterion groups & main
// ──────────────────────────────────────────────

criterion_group!(
    name = layer1_tokenizer;
    config = Criterion::default().sample_size(200);
    targets = bench_split_into_chunks, bench_mock_encode, bench_mock_decode, bench_real_encode_decode
);

criterion_group!(
    name = layer2_mock_pipeline;
    config = Criterion::default().sample_size(200);
    targets = bench_mock_translate
);

criterion_group!(
    name = layer3_onnx_inference;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30))
        .warm_up_time(std::time::Duration::from_secs(5));
    targets = bench_onnx_run_inference
);

criterion_group!(
    name = layer4_end_to_end;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(30))
        .warm_up_time(std::time::Duration::from_secs(5));
    targets = bench_end_to_end_translate
);

criterion_main!(
    layer1_tokenizer,
    layer2_mock_pipeline,
    layer3_onnx_inference,
    layer4_end_to_end
);
