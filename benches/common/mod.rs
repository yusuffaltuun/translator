#![allow(dead_code)]

pub fn real_paths() -> Option<(&'static str, &'static str, &'static str)> {
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

const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "machine",
    "neural",
    "translation",
    "system",
    "produces",
    "fluent",
    "natural",
    "language",
    "output",
    "sentence",
    "context",
    "overlap",
    "chunk",
    "batch",
    "tokenizer",
    "inference",
    "pipeline",
    "offline",
    "benchmark",
    "latency",
    "throughput",
    "memory",
    "allocation",
    "vector",
    "tensor",
    "decoder",
    "encoder",
    "attention",
    "cache",
    "sequence",
    "padding",
    "bucket",
    "window",
    "trimmer",
    "processes",
    "preserves",
    "meaning",
    "across",
    "boundaries",
    "reads",
    "writes",
];

pub fn generate_words(count: usize) -> String {
    let mut out = String::with_capacity(count * 7 + count / 13);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut in_sentence = true;
    for i in 0..count {
        if i > 0 {
            out.push(' ');
        }
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = ((state >> 33) % WORDS.len() as u64) as usize;
        let word = WORDS[idx];
        if in_sentence {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                out.push(first.to_ascii_uppercase());
                out.push_str(chars.as_str());
            }
            in_sentence = false;
        } else {
            out.push_str(word);
        }
        if (i + 1) % 13 == 0 {
            out.push('.');
            in_sentence = true;
        }
    }
    if !out.ends_with('.') {
        out.push('.');
    }
    out
}

pub fn mixed_corpus() -> String {
    let sizes = [300usize, 80, 250, 40, 400, 90, 260, 60];
    let paragraphs: Vec<String> = sizes.iter().map(|&n| generate_words(n)).collect();
    paragraphs.join("\n\n")
}
