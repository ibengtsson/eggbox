# Eggbox: active learning for global optimization

An interactive [marimo](https://marimo.io) notebook built as a hands-on tutorial for
graduate students who are curious about machine learning but new to it. It demonstrates
**active learning** — and specifically **Bayesian optimization** — on a deliberately tricky
toy problem.

## The idea

Imagine you can measure some quantity at any point `(x, y)` — the energy of a material, the
yield of a reaction, the score of a design — but **every measurement is expensive** (a lab
experiment, a long simulation). The landscape is **rough**, full of dips and bumps, with one
**global minimum** hiding somewhere. Checking a fine grid would take thousands of
measurements. Can we be smarter?

Active learning says yes: train a quick model on what you've measured, let it tell you
**where it's worth measuring next**, measure there, and repeat. The notebook shows this loop
pinning down the global minimum in a few dozen measurements instead of thousands.

## What the notebook covers

1. **The landscape** — a rough "funnel": periodic eggbox ripples (many local minima, after
   the eggbox from [MultiNest, arXiv:0809.3437](https://arxiv.org/pdf/0809.3437)) tilted
   inside a broad bowl, so the central well is the global minimum. Shown as a heatmap and an
   interactive 3D surface.
2. **Starting data** — a handful of measurements stuck in one corner, far from the optimum.
3. **The model and the loop** — a **deep ensemble** of small neural networks, where the
   ensemble's *disagreement* is the uncertainty estimate. A **Lower Confidence Bound**
   acquisition function, `a(x) = μ̂(x) − κ·σ̂(x)`, decides where to measure next, balancing
   exploitation (low predicted energy) against exploration (high uncertainty). Watch it
   march from the corner to the center, and scrub through the iterations.
4. **Is it better than guessing?** — a comparison against random search, with the same budget.

Everything is interactive: sliders control the landscape roughness, the starting data, the
exploration knob `κ`, and the measurement budget.

## Running it

The notebook declares its own dependencies inline (PEP 723), so [uv](https://docs.astral.sh/uv/)
handles the environment automatically:

```bash
# interactive editor
uv run --with marimo marimo edit eggbox_active_learning.py --sandbox

# read-only app (for presenting)
uv run --with marimo marimo run eggbox_active_learning.py --sandbox
```

## Sharing it

Export a fully interactive, **zero-install** version that runs in the browser via WebAssembly,
then host the resulting folder on any static host (e.g. GitHub Pages):

```bash
uvx marimo export html-wasm eggbox_active_learning.py -o dist --mode run   # clean app
uvx marimo export html-wasm eggbox_active_learning.py -o dist-code --mode edit  # code visible
```
