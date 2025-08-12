# Usage and Device Guide

This page summarizes the primary user-facing API and how to select devices (CPU, CUDA, MPS) consistently.

## Installation Smoke Test

```python
import supar
from supar import Parser
```

## Loading a pretrained parser

- Load by shortcut name (downloads to cache on first use):
```python
from supar import Parser
parser = Parser.load('dep-biaffine-en')
```

- Load from local path:
```python
parser = Parser.load('/path/to/model')
```

## Selecting device

- CPU (default):
```python
parser = Parser.load('dep-biaffine-en', device='cpu')
```

- CUDA (if available):
```python
parser = Parser.load('dep-biaffine-en', device='cuda')
# or later
parser.to('cuda')
```

- MPS (Apple Silicon):
```python
parser = Parser.load('dep-biaffine-en', device='mps')
# or later
parser.to('mps')
```

Notes:
- We instantiate the model on the target device before loading the state dict to ensure correct storage allocation.
- On MPS, we enforce float32 for robust kernels and rebuild embedding layers to avoid placeholder storages.
- Autocast is enabled on CUDA only.

## Prediction

```python
# Plain text with language hint so the tokenizer can be applied
parser.predict('She enjoys playing tennis.', lang='en')

# Already-tokenized input
parser.predict(['She', 'enjoys', 'playing', 'tennis', '.'])

# With probabilities
parser.predict('She enjoys playing tennis.', lang='en', prob=True)

# Override device just for prediction (moves once if needed)
parser.predict('She enjoys playing tennis.', lang='en', device='mps')
```

## Training and evaluation (unchanged)

The standard training and evaluation entry points remain the same, with improved device portability under the hood.
Refer to the docs of specific models under `models/` for task-specific arguments.

## Backwards compatibility

- Existing code using `Parser.load(...)` without a `device` argument continues to work.
- `Parser.to(device)` now performs a deeper relocation including embedding rebuilds on MPS and dtype enforcement.
- Dataloader and field tensors now respect MPS if available when `cuda` is not.

## Troubleshooting

- If you request `cuda` or `mps` and it is unavailable, SuPar falls back to CPU and logs a warning.
- For MPS performance, ensure macOS and PyTorch are up to date and consider a warm-up call after moving to MPS:
```python
parser.to('mps')  # optional warm-up is done on first real forward
```
