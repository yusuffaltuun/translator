use criterion::{black_box, criterion_group, criterion_main, Criterion};
use translator::translator::OverlapTrimmer;

fn bench_trimmer_exact_match(c: &mut Criterion) {
    let trimmer = OverlapTrimmer::default();
    let shared = "the neural machine translation system produces fluent natural language";
    let prev = format!("Sonuç olarak metin çevirisi tamamlandı. {shared}");
    let cur = format!("{shared} output that preserves the original meaning across boundaries");
    c.bench_function("trimmer/exact_match", |b| {
        b.iter(|| black_box(trimmer.trim(black_box(&prev), black_box(&cur))))
    });
}

fn bench_trimmer_fuzzy_match(c: &mut Criterion) {
    let trimmer = OverlapTrimmer::default();
    let prev =
        "The system successfully produced translation engines generate fluent natural output";
    let cur =
        "translatoin engines generate fluent natural output that was faithfully rendered into Turkish";
    c.bench_function("trimmer/fuzzy_match", |b| {
        b.iter(|| black_box(trimmer.trim(black_box(prev), black_box(cur))))
    });
}

fn bench_trimmer_no_match(c: &mut Criterion) {
    let trimmer = OverlapTrimmer::default();
    let prev = "The weather today is sunny and warm with a gentle breeze across the valley";
    let cur =
        "Quantum physics describes the behaviour of subatomic particles with remarkable precision";
    c.bench_function("trimmer/no_match_fallback", |b| {
        b.iter(|| black_box(trimmer.trim(black_box(prev), black_box(cur))))
    });
}

criterion_group!(
    name = bench_overlap_trimmer;
    config = Criterion::default();
    targets = bench_trimmer_exact_match, bench_trimmer_fuzzy_match, bench_trimmer_no_match
);

criterion_main!(bench_overlap_trimmer);
