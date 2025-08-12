# Unified API Guide

This page provides a concise, single-page API for SuPar, centered on the unified `supar.Parser` interface. It covers installation, model loading, device selection (`cpu`/`cuda`/`mps`), prediction, evaluation, training, saving, and expected inputs/outputs.

## Install and import

```bash
pip install -U supar
# or from source
pip install -U git+https://github.com/yzhangcs/parser
```

```python
from supar import Parser
```

## Load a pretrained parser

- By shortcut name (downloads/caches on first use):

```python
parser = Parser.load('dep-biaffine-en')
```

- From local path (directory containing a saved `model` file):

```python
parser = Parser.load('/path/to/model')
```

- Specify device at load time:

```python
# device can be 'cpu', 'cuda', 'mps', or torch.device(...)
parser = Parser.load('dep-biaffine-en', device='cuda')
```

Notes:
- If the requested device is not available, the parser falls back to CPU and logs a warning.
- When using `mps` (Apple Silicon), the model runs in float32 for robust kernels and rebuilds embedding layers for correct allocation.

## Device control

```python
# Move an existing parser to a device (with optional dtype)
parser = parser.to('cuda')
parser = parser.to('mps')              # enforces float32 by default
parser = parser.to('cpu', dtype=None)  # CPU, leave dtype as-is
```

Autocast is enabled only on CUDA during prediction/evaluation when `amp=True`. On MPS, autocast is disabled and float32 is enforced.

## Prediction

SuPar supports multiple input forms.

```python
# 1) Plain text (tokenized internally if you provide a language)
parser.predict('She enjoys playing tennis.', lang='en')

# 2) Already-tokenized sentence
parser.predict(['She', 'enjoys', 'playing', 'tennis', '.'])

# 3) From file (e.g., .txt or CoNLL-style)
parser.predict('/path/to/data.txt', pred='pred.conllx')

# Return probabilities
parser.predict('She enjoys playing tennis.', lang='en', prob=True)

# Override device just for prediction
parser.predict('She enjoys playing tennis.', lang='en', device='mps')
```

Arguments (common):
- `data`: str or iterable (file path, plain text, token list, or pre-tokenized structured input depending on the model).
- `pred`: optional output filename for predictions.
- `lang`: language hint for tokenizer (e.g., `'en'`).
- `prob`: include probabilities in the returned dataset.
- `batch_size`, `buckets`, `workers`: throughput-related knobs.
- `device`: optional per-call device move; otherwise uses the parser’s current device.
- `amp`: enable CUDA autocast during prediction.

Returns:
- A `supar.utils.Dataset` containing predicted sentences and, when requested, probabilities. If `cache=True`, predictions are streamed to disk and the function returns `None`.

## Evaluation

```python
metric = parser.evaluate('/path/to/gold.conllx')
print(metric)
```

Key args:
- `data`: filename or iterable of examples.
- `batch_size`, `buckets`, `workers`: throughput.
- `amp`: enable CUDA autocast during evaluation.

Returns:
- A task-specific `Metric` summarizing accuracy (e.g., UAS/LAS for dependency parsing).

## Training (programmatic)

```python
parser.train(
    train='train.conllx',
    dev='dev.conllx',
    test='test.conllx',
    epochs=40,
    patience=10,
    batch_size=5000,
    update_steps=1,
    buckets=32,
    workers=0,
    amp=False,
    cache=False,
)
```

Notes:
- Distributed training is supported via torch.distributed; per-step sync is handled internally.
- Mixed precision (`amp=True`) accelerates training on CUDA.

## Saving and loading

```python
# Save current model state (weights + transform)
parser.save('model')

# Load from shortcut or local path
parser = Parser.load('dep-biaffine-en')
parser = Parser.load('/path/to/model')
```

## Frequently used model shortcuts

- Dependency: `dep-biaffine-<lang>`, `dep-crf2o-<lang>`, `dep-biaffine-roberta-en`, `dep-biaffine-xlmr`, etc.
- Constituency: `con-crf-<lang>`, `con-crf-roberta-en`, `con-crf-xlmr`, etc.
- Semantic dependency: `sdp-biaffine-<lang>`, `sdp-vi-<lang>`, `sdp-vi-roberta-en`, etc.

See the README for the full list and performance tables.

## Tips and troubleshooting

- If you request `cuda`/`mps` but it’s unavailable, SuPar falls back to CPU and logs a warning.
- For best MPS performance, ensure macOS and PyTorch are up-to-date. A warm-up pass is optional; real calls will initialize kernels lazily.
- Ensure any file-based inputs conform to the expected task format (e.g., CoNLL-X/CoNLL-U variants for dependency parsing).

## Minimal end-to-end example

```python
from supar import Parser

# Load on GPU if available, else fallback
parser = Parser.load('dep-biaffine-en', device='cuda')

# Predict with probabilities
dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True)
print(dataset[0])

# Evaluate
metric = parser.evaluate('ptb/test.conllx')
print(metric)

# Save
parser.save('model')
```


