#[path = "common/mod.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use translator::errors::TranslatorError;
use translator::traits::{InferenceEngine, Tokenizer};
use translator::translator::Translator;

#[derive(Clone, Copy)]
struct PipelineMockTokenizer;

fn word_id(word: &str) -> i64 {
    word.bytes()
        .fold(0i64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as i64))
        .rem_euclid(50_000)
        + 1
}

impl Tokenizer for PipelineMockTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TranslatorError> {
        Ok(text.split_whitespace().map(word_id).collect())
    }

    fn decode(&self, ids: &[i64]) -> Result<String, TranslatorError> {
        Ok(ids
            .iter()
            .map(|id| format!("t{id}"))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, TranslatorError> {
        Ok(text.split_whitespace().count())
    }
}

#[derive(Clone, Copy)]
struct PipelineMockEngine {
    max_source: usize,
}

impl InferenceEngine for PipelineMockEngine {
    fn max_source_len(&self) -> usize {
        self.max_source
    }

    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<i64>, TranslatorError> {
        Ok(input_ids.to_vec())
    }

    fn run_inference_batch(
        &mut self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<Vec<i64>>, TranslatorError> {
        Ok(input_ids
            .iter()
            .zip(attention_mask)
            .map(|(row, mask)| {
                row.iter()
                    .zip(mask)
                    .filter(|&(_, &m)| m == 1)
                    .map(|(&t, _)| t)
                    .collect()
            })
            .collect())
    }
}

fn bench_mock_pipeline_overhead(c: &mut Criterion) {
    let text = common::generate_words(1000);
    let mut translator = Translator::new(
        PipelineMockTokenizer,
        PipelineMockEngine { max_source: 512 },
    );
    translator.probe_batch_size().unwrap();

    let mut group = c.benchmark_group("mock_pipeline");
    group.throughput(Throughput::Elements(1000));
    group.bench_function("orchestration/1000_words", |b| {
        b.iter(|| {
            let result = translator.translate(black_box(&text)).unwrap();
            black_box(result)
        })
    });
    group.finish();
}

criterion_group!(
    name = bench_mock_pipeline;
    config = Criterion::default().sample_size(200);
    targets = bench_mock_pipeline_overhead
);

criterion_main!(bench_mock_pipeline);
