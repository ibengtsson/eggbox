# Eggbox: active learning for global optimization

A hands-on tutorial for graduate students who are curious about machine learning but new to
it. It's an interactive [marimo](https://marimo.io) notebook plus a worksheet, and it takes
about 40 minutes: **~30 minutes working through five tasks, then a group discussion.**

## 👋 Students — start here

1. **Open the notebook:** **[ibengtsson.github.io/eggbox](https://ibengtsson.github.io/eggbox/)**
   — it runs entirely in your browser, nothing to install. Give it a minute to boot.
2. **Open the worksheet:** **[`WORKSHEET.md`](WORKSHEET.md)** — the four tasks and the
   questions to answer. Read it alongside the notebook, and keep pen and paper (or a scratch
   file) next to you for your answers.
3. Work through Tasks 1 → 5 in order. Each has a 💡 hint you can unfold if you get stuck, and
   ends with a 🔍 Reveal box — **write your answer before you open that one.**

Two practical notes: every ▶ Run button retrains the model from scratch, so it takes a few
seconds (longer in the browser version — read the next question while you wait), and nothing
you do is saved, so write your answers down as you go.

The optional code task at the end needs a notebook where the code is visible: use
**[the /edit/ version](https://ibengtsson.github.io/eggbox/edit/)** instead.

Prefer to run it on your own machine? It's faster than the browser version and lets you edit
anything — see [Running it on your own machine](#-running-it-on-your-own-machine) below.

## The idea

Imagine you can measure some quantity at any point `(x, y)` — the energy of a material, the
yield of a reaction, the score of a design — but **every measurement is expensive** (a lab
experiment, a long simulation). The landscape is **rough**, full of dips and bumps, with one
**global minimum** hiding somewhere. Checking a fine grid would take thousands of
measurements. Can we be smarter?

Active learning says yes: train a quick model on what you've measured, let it tell you
**where it's worth measuring next**, measure there, and repeat. When the goal is specifically
to find the optimum, the loop is called **Bayesian optimization** — and the tutorial asks you
to find out for yourself whether it beats simply guessing.

## What you'll work through

1. **The landscape** — a rough "funnel": periodic eggbox ripples (many local minima, after
   the eggbox from [MultiNest, arXiv:0809.3437](https://arxiv.org/pdf/0809.3437)) tilted
   inside a broad bowl, so the central well is the global minimum. Shown as a heatmap and an
   interactive 3D surface. *(Task 1: why is this hard?)*
2. **Starting data** — a handful of measurements stuck in one corner, far from the optimum.
3. **The model and the loop** — a **deep ensemble** of small neural networks, where the
   ensemble's *disagreement* is the uncertainty estimate. A **Lower Confidence Bound**
   acquisition function, `a(x) = μ̂(x) − κ·σ̂(x)`, decides where to measure next, balancing
   exploitation (low predicted energy) against exploration (high uncertainty).
   *(Task 2: what does the model actually know — and is its uncertainty the same as its
   error?)*
4. **The exploration dial** — the same loop run at κ = 0, 1.5 and 4 from identical starting
   data, so the trade-off is measured rather than asserted. *(Task 3: which κ wins, and can
   you trust one run?)*
5. **Is it better than guessing?** — a fixed-budget challenge against random search, plus a
   three-seed check that shows how much of any single result is luck. *(Task 4: beat the
   baseline.)*
6. **What if the trend lies?** — the same eggbox, but the bowl now bottoms out at a decoy and
   the real optimum hides in a pocket no smooth trend points at. Greed walks into the trap and
   stays there; only the curious setting gets out. *(Task 5.)*

## 🧑‍🏫 Instructors

The worksheet is the student handout. The companion **instructor notes** — expected answers,
common wrong turns, measured timings and a script for the wrap-up discussion — are
deliberately **not in this repository**, since it's public and the students are pointed here.
Ask the maintainer for a copy.

Two things worth knowing before you run the session: the browser build is several times
slower than running locally (budget extra time for each ▶ Run, or have students run it
natively), and the outcomes are genuinely seed-noisy — that noise is the subject of Tasks 3
and 4 rather than a defect to tune away.

## 🖥 Running it on your own machine

Optional, but it runs several times faster than the browser version, and it's the only way to
keep the changes you make — edits on the `/edit/` page vanish when you reload. There's no
Python environment to set up.

**1. Install [uv](https://docs.astral.sh/uv/)** (a one-line installer; it manages Python for
you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                 # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"      # Windows
```

**2. Clone this repo and open the notebook:**

```bash
git clone https://github.com/ibengtsson/eggbox.git
cd eggbox
uv run --with marimo marimo edit eggbox_active_learning.py --sandbox
```

It opens in your browser. The first launch takes a minute while uv downloads numpy, plotly
and scikit-learn; after that it starts instantly.

You don't need a virtualenv or a `requirements.txt`: the notebook lists its own dependencies
at the top of the file (PEP 723), and `--sandbox` builds a throwaway environment from them.

### Editing the code

marimo notebooks are **plain Python files**, and cells re-run automatically when anything
they depend on changes — so change a line, run the cell, and every plot downstream updates
itself. Execution order comes from which cell uses which variable, not from top to bottom.

The one cell the optional task asks you to edit is marked with a 👇 comment banner; the rest
is fair game too. To throw your edits away and start over:

```bash
git checkout eggbox_active_learning.py
```

Swap `marimo edit` for `marimo run` to get the same notebook with the code hidden — the view
you get on the website.

## Publishing it

The notebook exports to a fully interactive, **zero-install** page that runs in the browser
via WebAssembly, hostable on any static host:

```bash
uvx marimo export html-wasm eggbox_active_learning.py -o dist --mode run        # clean app
uvx marimo export html-wasm eggbox_active_learning.py -o dist/edit --mode edit  # code visible
```

`.github/workflows/deploy.yml` does exactly this on every push to `main`, publishing the app
at the site root and the editable copy at `/edit/`.
