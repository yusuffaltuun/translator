#[path = "common/mod.rs"]
mod common;

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use translator::translator::DefaultTranslator;

static TRANSLATOR: OnceLock<Mutex<DefaultTranslator>> = OnceLock::new();

fn translator() -> &'static Mutex<DefaultTranslator> {
    TRANSLATOR.get_or_init(|| {
        let (enc, dec, tok) =
            common::real_paths().expect("ONNX models not found; cannot run bench_end_to_end");
        let mut translator = DefaultTranslator::load_model(enc, dec, tok).unwrap();
        translator.probe_batch_size().unwrap();
        Mutex::new(translator)
    })
}

fn bench_short(c: &mut Criterion) {
    let text = common::generate_words(100);
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(100));
    group.bench_function("short_100_words", |b| {
        b.iter(|| {
            let out = translator()
                .lock()
                .unwrap()
                .translate(black_box(&text))
                .unwrap();
            black_box(out)
        })
    });
    group.finish();
}

fn bench_medium(c: &mut Criterion) {
    let text = common::generate_words(1000);
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1000));
    group.bench_function("medium_1000_words", |b| {
        b.iter(|| {
            let out = translator()
                .lock()
                .unwrap()
                .translate(black_box(&text))
                .unwrap();
            black_box(out)
        })
    });
    group.finish();
}

fn bench_long(c: &mut Criterion) {
    let text = common::generate_words(5000);
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));
    group.warm_up_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(5000));
    group.bench_function("long_5000_words", |b| {
        b.iter(|| {
            let out = translator()
                .lock()
                .unwrap()
                .translate(black_box(&text))
                .unwrap();
            black_box(out)
        })
    });
    group.finish();
}

criterion_group!(
    name = bench_end_to_end;
    config = Criterion::default();
    targets = bench_short, bench_medium, bench_long
);

criterion_main!(bench_end_to_end);
