#[path = "common/mod.rs"]
mod common;

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use translator::traits::{InferenceEngine, Tokenizer};
use translator::translator::{build_batches, Chunker, HuggingFaceTokenizer, OnnxInferenceEngine};

const PAD_TOKEN_ID: i64 = 57059;
const SAMPLE_TEXT: &str =
    "The machine translation system processes each sentence and preserves the original meaning \
     across chunk boundaries without losing context.";

static ENGINE: OnceLock<Mutex<OnnxInferenceEngine>> = OnceLock::new();

fn engine() -> &'static Mutex<OnnxInferenceEngine> {
    ENGINE.get_or_init(|| {
        let (enc, dec, _) =
            common::real_paths().expect("ONNX models not found; cannot run bench_onnx_batching");
        Mutex::new(OnnxInferenceEngine::load(enc, dec).unwrap())
    })
}

static TOKENIZER: OnceLock<HuggingFaceTokenizer> = OnceLock::new();

fn tokenizer() -> &'static HuggingFaceTokenizer {
    TOKENIZER.get_or_init(|| {
        let (_, _, tok) =
            common::real_paths().expect("tokenizer not found; cannot run bench_onnx_batching");
        HuggingFaceTokenizer::from_file(tok).unwrap()
    })
}

fn bench_onnx_fixed_batch_sizes(c: &mut Criterion) {
    let ids = tokenizer().encode(SAMPLE_TEXT).unwrap();
    let mut group = c.benchmark_group("onnx_batching");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(3));

    for batch_size in [1usize, 2, 4, 8] {
        let input_ids = vec![ids.clone(); batch_size];
        let attention_mask = vec![vec![1i64; ids.len()]; batch_size];
        group.bench_function(format!("fixed_batch_size_{batch_size}"), move |b| {
            b.iter(|| {
                let mut eng = engine().lock().unwrap();
                let out = eng
                    .run_inference_batch(black_box(&input_ids), black_box(&attention_mask))
                    .unwrap();
                black_box(out)
            })
        });
    }

    group.finish();
}

fn mask_for(real_len: usize, max_len: usize) -> Vec<i64> {
    let mut mask = vec![1i64; real_len];
    mask.resize(max_len, 0i64);
    mask
}

type PaddedBatch = (Vec<Vec<i64>>, Vec<Vec<i64>>);

fn build_batches_unsorted(
    chunks: &[(usize, Vec<i64>)],
    batch_size: usize,
    pad_id: i64,
) -> Vec<PaddedBatch> {
    let batch_size = batch_size.max(1);
    let mut batches = Vec::new();
    for group in chunks.chunks(batch_size) {
        let max_len = group.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(group.len());
        let mut attention_mask = Vec::with_capacity(group.len());
        for (_, ids) in group {
            let mut padded = ids.clone();
            padded.resize(max_len, pad_id);
            input_ids.push(padded);
            attention_mask.push(mask_for(ids.len(), max_len));
        }
        batches.push((input_ids, attention_mask));
    }
    batches
}

fn bench_length_bucketing(c: &mut Criterion) {
    let corpus = common::mixed_corpus();
    let chunker = Chunker::default();
    let chunks = chunker
        .split(&corpus, |piece| tokenizer().count_tokens(piece))
        .unwrap();
    let encoded: Vec<(usize, Vec<i64>)> = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (index, tokenizer().encode(chunk).unwrap()))
        .collect();
    let chunk_count = encoded.len();

    let mut group = c.benchmark_group("length_bucketing_batch4");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(chunk_count as u64));

    group.bench_function("bucketing_enabled", |b| {
        b.iter(|| {
            let batches = build_batches(black_box(&encoded), 4, PAD_TOKEN_ID);
            let mut eng = engine().lock().unwrap();
            let mut out = Vec::new();
            for batch in &batches {
                out.push(
                    eng.run_inference_batch(&batch.input_ids, &batch.attention_mask)
                        .unwrap(),
                );
            }
            black_box(out)
        })
    });

    group.bench_function("bucketing_disabled", |b| {
        b.iter(|| {
            let batches = build_batches_unsorted(black_box(&encoded), 4, PAD_TOKEN_ID);
            let mut eng = engine().lock().unwrap();
            let mut out = Vec::new();
            for (input_ids, attention_mask) in &batches {
                out.push(eng.run_inference_batch(input_ids, attention_mask).unwrap());
            }
            black_box(out)
        })
    });

    group.finish();
}

criterion_group!(
    name = bench_onnx_batching;
    config = Criterion::default();
    targets = bench_onnx_fixed_batch_sizes, bench_length_bucketing
);

criterion_main!(bench_onnx_batching);
