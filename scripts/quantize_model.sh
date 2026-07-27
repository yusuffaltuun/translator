#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# quantize_model.sh — INT8 dynamic quantization for MarianMT ONNX models
#
# Quantizes both encoder_model.onnx and decoder_model.onnx to INT8,
# reducing model sizes by ~4× with <1% BLEU score loss.
# Requires Python 3.10+ and onnxruntime >= 1.17.
#
# Usage:
#   ./scripts/quantize_model.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

MODELS_DIR="${1:-models}"

echo "── INT8 Dynamic Quantization ──"
echo "  Models directory: $MODELS_DIR"
echo ""

python3 -c "
import sys, os

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print('Error: onnxruntime not installed.', file=sys.stderr)
    print('  pip install onnxruntime>=1.17', file=sys.stderr)
    sys.exit(1)

models_dir = sys.argv[1]

models = [
    ('encoder_model.onnx', 'encoder_model_int8.onnx'),
    ('decoder_model.onnx', 'decoder_model_int8.onnx'),
]

for input_name, output_name in models:
    input_path = os.path.join(models_dir, input_name)
    output_path = os.path.join(models_dir, output_name)

    if not os.path.exists(input_path):
        print(f'  Skipping {input_name} (not found)')
        continue

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f'Quantizing {input_name} ({input_size:.0f} MB) -> {output_name} ...')
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  Before: {input_size:.0f} MB  After: {output_size:.0f} MB')
    print()

print('Done. Place quantized models alongside the originals in models/.')
" "$MODELS_DIR"

echo ""
echo "Quantized models generated in $MODELS_DIR."
echo "Update Rust code to load *_int8.onnx when available."
