#[path = "common/mod.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use translator::traits::Tokenizer;
use translator::translator::{Chunker, HuggingFaceTokenizer};

fn word_proxy_estimator(piece: &str) -> Result<usize, std::convert::Infallible> {
    Ok(piece.split_whitespace().count())
}

fn bench_chunker_word_proxy(c: &mut Criterion) {
    let text = common::generate_words(1000);
    let chunker = Chunker::default();
    c.bench_function("chunker/split_word_proxy/1000_words", |b| {
        b.iter(|| {
            let chunks = chunker
                .split(black_box(&text), word_proxy_estimator)
                .unwrap();
            black_box(chunks)
        })
    });
}

fn bench_chunker_real_tokens(c: &mut Criterion) {
    let (_, _, tok) = match common::real_paths() {
        Some(paths) => paths,
        None => return,
    };
    let tokenizer = HuggingFaceTokenizer::from_file(tok).unwrap();
    let text = common::generate_words(1000);
    let chunker = Chunker::default();
    c.bench_function("chunker/split_real_tokenizer/1000_words", |b| {
        b.iter(|| {
            let chunks = chunker
                .split(black_box(&text), |piece| tokenizer.count_tokens(piece))
                .unwrap();
            black_box(chunks)
        })
    });
}

fn bench_real_encode(c: &mut Criterion) {
    let (_, _, tok) = match common::real_paths() {
        Some(paths) => paths,
        None => return,
    };
    let tokenizer = HuggingFaceTokenizer::from_file(tok).unwrap();
    let text = common::generate_words(32);
    c.bench_function("tokenizer/encode/sentence_32_words", |b| {
        b.iter(|| {
            let ids = tokenizer.encode(black_box(&text)).unwrap();
            black_box(ids)
        })
    });
}

fn bench_real_decode(c: &mut Criterion) {
    let (_, _, tok) = match common::real_paths() {
        Some(paths) => paths,
        None => return,
    };
    let tokenizer = HuggingFaceTokenizer::from_file(tok).unwrap();
    let text = common::generate_words(32);
    let ids = tokenizer.encode(&text).unwrap();
    c.bench_function("tokenizer/decode/tokens_32_words", |b| {
        b.iter(|| {
            let out = tokenizer.decode(black_box(&ids)).unwrap();
            black_box(out)
        })
    });
}

criterion_group!(
    name = bench_text_handling;
    config = Criterion::default().sample_size(200);
    targets = bench_chunker_word_proxy, bench_chunker_real_tokens, bench_real_encode, bench_real_decode
);

criterion_main!(bench_text_handling);
