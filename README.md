# PixelVAEPytorch

PyTorch translation of the original `PixelVAE` repository.

## What is included

- `mnist_model.py`
  - Theano MNIST PixelVAE port
  - `cond_z_bias` and `upsample_z_conv` decoders
  - blind-spot-free vertical/horizontal PixelCNN stacks
- `pixelvae_model.py`
  - TensorFlow one-level and two-level PixelVAE port
  - `Enc1` / `Dec1` / `Enc2` / `Dec2`
  - `EncFull` / `DecFull`
  - discrete input embeddings, masked residual blocks, and staged KL terms
- `ops.py`
  - masked convolutions
  - weight-normalized linear/conv helpers
  - subpixel upsampling
- `configs.py`
  - original preset settings:
    `mnist_256`, `32px_small`, `32px_big`, `64px_small`, `64px_big`, `64px_big_onelevel`

## Install

```bash
uv sync
```

Or:

```bash
pip install -e .
```

## Quick checks

From the repository root:

```bash
uv run --active smoke_test.py
uv run --active pixelvae.py --settings mnist_256
```

After installation:

```bash
pixelvae-smoke-test
pixelvae-sample --settings mnist_256
```

## MNIST scripts

Train:

```bash
uv run --active mnist_pixelvae_train.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16
```

Evaluate from a checkpoint:

```bash
uv run --active mnist_pixelvae_evaluate.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -w path/to/checkpoint.pt
```

This writes a PNG sample grid by default to `runs/mnist/eval_samples.png`.

Notes:

- `mnist_data.py` downloads and parses the original IDX gzip files directly.
- Use `--fake-data` for a fast smoke run without downloads.
