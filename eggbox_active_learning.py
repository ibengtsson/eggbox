# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Finding a needle in a rough landscape 🥚🔎
    ## An introduction to **active learning**

    Imagine you can *measure* some quantity at any point `(x, y)` — the energy of a material,
    the yield of a reaction, the score of a design — but **each measurement is expensive**
    (a lab experiment, a long simulation). The landscape is **rough**: full of dips and
    bumps, with one **global minimum** hiding somewhere that you want to find. Checking every
    point on a fine grid would take thousands of measurements. Can we be smarter?

    **Active learning** says yes: train a quick model on what you've measured so far, let it
    tell you **where it's worth measuring next**, measure there, and repeat. When the goal is
    specifically *finding the optimum*, this loop is also called **Bayesian optimization**.

    Our playground is the **"eggbox"** — a bumpy, periodic surface from the MultiNest paper
    ([arXiv:0809.3437](https://arxiv.org/pdf/0809.3437)) — tilted into a broad bowl so that
    one well, at the center, is the deepest. We'll start stuck in a corner and try to hunt
    the center down.

    /// admonition | 📋 Work through this with the worksheet
    This notebook has **five tasks**. Each one is marked with a 🎯 box and matches a numbered
    task on your worksheet — do the steps, then write your answer on the sheet. No coding
    required (there's an optional code task at the very bottom if you finish early).

    Each task also has a **💡 Hint** you can unfold if you get stuck, and the sections end
    with a **🔍 Reveal** box — don't open that one until you've written your answer down.
    ///
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell
def _():
    import warnings

    from plotly.subplots import make_subplots
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    return ConvergenceWarning, MLPRegressor, make_subplots, warnings


@app.cell
def _():
    def tidy(fig, legend=False, left=60, right=26, top=54, bottom=None):
        """One consistent layout, so nothing lands on top of anything else.

        Plotly parks its toolbar in the top-right corner and centres the title right under
        it, and a legend placed above the axes joins the same pile. Anchoring the title to
        the left, standing the toolbar up as a vertical strip and dropping the legend below
        the plot keeps the three of them out of each other's way.
        """
        fig.update_layout(
            title=dict(x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)),
            modebar=dict(orientation="v"),
            margin=dict(l=left, r=right, t=top,
                        b=bottom if bottom is not None else (92 if legend else 48)),
        )
        if legend:
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.30, x=0, xanchor="left"),
            )
        return fig

    return (tidy,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The landscape: a rough funnel

    Here's the surface we want to minimize. Think of it as an **energy landscape** — low is
    good, and we're hunting for the lowest point of all.

    $$ E(x, y) = \underbrace{-r\,\cos\!\tfrac{x-c}{2}\cos\!\tfrac{y-c}{2}}_{\text{eggbox ripples}}
       \;+\; \underbrace{s\,\frac{(x-c)^2 + (y-c)^2}{R^2}}_{\text{broad bowl}} $$

    The **ripples** carve a grid of wells — lots of **local minima** to get trapped in. The
    **bowl** gently tilts the whole thing toward the middle, so the **central well is the
    global minimum** (marked 🟢). The `roughness` slider sets how deep the local wells are.
    Below about **0.9** the bowl wins everywhere and the surface has a *single* minimum — no
    traps at all, and the problem is trivial. Above it the ripples start carving out
    genuine local minima: 9 of them, and from roughness ≈ 2.2 onward the full 13.
    """)
    return


@app.cell
def _(mo):
    roughness_slider = mo.ui.slider(
        start=0.0, stop=4.0, value=3.0, step=0.5,
        label="roughness (depth of the local wells)",
    )
    roughness_slider
    return (roughness_slider,)


@app.cell
def _(np, roughness_slider):
    # A rough "funnel": eggbox ripples (local minima) inside a broad bowl whose bottom
    # — the global minimum — sits at the center of the domain.
    DOMAIN_MIN = 0.0
    DOMAIN_MAX = 10.0 * np.pi
    CENTER = DOMAIN_MAX / 2.0
    BOWL_RADIUS = DOMAIN_MAX / 2.0
    BOWL_STRENGTH = 6.0
    # A run "has found the global well" once its best measurement is BOTH inside the
    # central ripple cell AND within this much energy of the true minimum.
    # The rival wells sit 1.92 above the global minimum (a roughness-independent bowl
    # offset), so any tolerance below that already pins the central cell on its own —
    # measured, no point anywhere passes the energy test from outside the disc. The disc is
    # therefore a safety belt, and starts doing real work only if BOWL_STRENGTH is lowered
    # or FOUND_TOL raised past 1.92.
    WELL_RADIUS = 2.0 * np.pi   # half a ripple period — the central well's own cell
    FOUND_TOL = 0.9             # ≈ half the 1.92 gap to the second-best well

    def funnel(x, y):
        """The true landscape. Low = good; the global minimum is the central well."""
        ripple = -roughness_slider.value * (
            np.cos(0.5 * (x - CENTER)) * np.cos(0.5 * (y - CENTER))
        )
        r2 = (x - CENTER) ** 2 + (y - CENTER) ** 2
        bowl = BOWL_STRENGTH * r2 / BOWL_RADIUS ** 2
        return ripple + bowl

    return CENTER, DOMAIN_MAX, DOMAIN_MIN, FOUND_TOL, WELL_RADIUS, funnel


@app.cell
def _(CENTER, DOMAIN_MAX, DOMAIN_MIN, funnel, go, mo, np, tidy):
    # Preview of the landscape (updates live with the roughness slider).
    funnel_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 160)
    fxx, fyy = np.meshgrid(funnel_axis, funnel_axis)
    funnel_z = funnel(fxx, fyy)

    fig_funnel = go.Figure(
        go.Surface(
            x=funnel_axis, y=funnel_axis, z=funnel_z,
            colorscale="Viridis", reversescale=True,
            colorbar=dict(title="energy", thickness=12),
        )
    )
    fig_funnel.add_trace(
        go.Scatter3d(
            x=[CENTER], y=[CENTER], z=[float(funnel(CENTER, CENTER))],
            mode="markers",
            marker=dict(color="lime", size=6, line=dict(color="black", width=1)),
            name="global minimum",
        )
    )
    fig_funnel.update_layout(
        title="The true landscape (we only get to see this because it's a demo)",
        width=680, height=480, showlegend=False,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="energy"),
    )
    tidy(fig_funnel, left=0, right=60, bottom=0)   # room for the colourbar under the toolbar

    fig_funnel_top = go.Figure(go.Heatmap(
        x=funnel_axis, y=funnel_axis, z=funnel_z,
        colorscale="Viridis", reversescale=True,
        colorbar=dict(title="energy", thickness=12),
    ))
    fig_funnel_top.update_layout(
        title="Top-down view — dark = deep", width=540, height=480,
        xaxis_title="x", yaxis_title="y",
    )
    fig_funnel_top.update_yaxes(scaleanchor="x", scaleratio=1)
    tidy(fig_funnel_top, right=90)

    mo.ui.tabs({"3D surface": fig_funnel, "top-down map": fig_funnel_top})
    return funnel_axis, funnel_z


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            ### 🎯 Task 1 — Why is this hard?  *(~4 min)*

            1. Drag the **roughness** slider above from `0` to `4` and back, and watch the
               surface. Spin the 3D view around; the *top-down map* tab is often easier to
               count on.
            2. Put roughness back to **3** before moving on.

            Then answer **Q1.1–Q1.3** on the worksheet: how many separate dips are there,
            what happens if you just walk downhill from the corner, and what are the odds
            that a single random guess lands in the right dip?
            """),
            kind="info",
        ),
        mo.accordion({
            "💡 Hint (Task 1)": mo.md(
                "Counting along an edge won't work — the ripples flatten to nothing there. "
                "Use the *top-down map* instead: the dips form a **checkerboard**, dark "
                "cells alternating with bright ones, spaced $2\\pi$ apart in $x$ and in "
                "$y$. Count the dark cells and you should land somewhere around a dozen. "
                "For Q1.3, if there are $N$ dips and only one is the right one, a blind "
                "guess has about a $1/N$ chance."
            ),
        }),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Start with data in one corner

    Suppose all we've measured so far sits in the **bottom-left corner** — maybe that's where
    we happened to start looking. The global minimum is far away at the center, and we have
    no data near it. The sliders set how much of the corner we sampled and how many points
    we have.

    The **random seed** just picks *which* random points we got. Keep it at 0 for now — you
    will change it later, deliberately, to check whether your conclusions survive.
    """)
    return


@app.cell
def _(mo):
    n_points_slider = mo.ui.slider(
        start=5, stop=100, value=25, step=5, label="number of initial points"
    )
    # The corner slider stops at 0.35 on purpose. Two separate thresholds sit above it:
    # geometrically the corner box first touches the central well's disc at f = (5−√2)/10 =
    # 0.359 (at the 0.35 cap its nearest point to the centre is 6.66 against a WELL_RADIUS of
    # 6.28 — ~6% of slack), but touching the disc is harmless on its own, because the "found"
    # test also demands an energy within FOUND_TOL of the true minimum. The premise actually
    # breaks near f ≈ 0.47, where the starting data first contains the answer (measured over
    # 1000 seeds at 25 points: 0.0% of seeds "found at 0" up to f = 0.46, 1.0% at 0.47,
    # 4.1% at 0.48, 16.4% at 0.50).
    corner_frac_slider = mo.ui.slider(
        start=0.1, stop=0.35, value=0.25, step=0.05,
        label="corner size (fraction of domain)",
    )
    seed_slider = mo.ui.slider(start=0, stop=20, value=0, step=1, label="random seed")

    sliders = mo.vstack([n_points_slider, corner_frac_slider, seed_slider])
    return corner_frac_slider, n_points_slider, seed_slider, sliders


@app.cell
def _(DOMAIN_MAX, DOMAIN_MIN, corner_frac_slider, n_points_slider, np, seed_slider):
    # Sample the initial measurement locations uniformly inside the bottom-left corner.
    rng = np.random.default_rng(seed_slider.value)

    corner_hi = DOMAIN_MIN + corner_frac_slider.value * (DOMAIN_MAX - DOMAIN_MIN)
    train_x = rng.uniform(DOMAIN_MIN, corner_hi, size=n_points_slider.value)
    train_y = rng.uniform(DOMAIN_MIN, corner_hi, size=n_points_slider.value)
    return corner_hi, train_x, train_y


@app.cell
def _(
    CENTER,
    DOMAIN_MIN,
    corner_hi,
    funnel_axis,
    funnel_z,
    go,
    mo,
    sliders,
    tidy,
    train_x,
    train_y,
):
    fig_data = go.Figure()
    fig_data.add_trace(
        go.Heatmap(
            x=funnel_axis, y=funnel_axis, z=funnel_z,
            colorscale="Viridis", reversescale=True, opacity=0.6,
            colorbar=dict(title="energy", thickness=12),
        )
    )
    fig_data.add_shape(
        type="rect", x0=DOMAIN_MIN, y0=DOMAIN_MIN, x1=corner_hi, y1=corner_hi,
        line=dict(color="white", width=1.5, dash="dash"),
    )
    fig_data.add_trace(
        go.Scatter(
            x=train_x, y=train_y, mode="markers",
            marker=dict(color="red", size=7, line=dict(color="white", width=1)),
            name="initial data",
        )
    )
    fig_data.add_trace(
        go.Scatter(
            x=[CENTER], y=[CENTER], mode="markers",
            marker=dict(color="lime", size=13, symbol="diamond",
                        line=dict(color="black", width=1)),
            name="global minimum",
        )
    )
    fig_data.update_layout(
        title="What we have measured so far",
        xaxis_title="x", yaxis_title="y",
        width=580, height=520, showlegend=False,
    )
    fig_data.update_yaxes(scaleanchor="x", scaleratio=1)
    tidy(fig_data, right=90)

    # Sliders sit directly above the plot so changes take effect immediately.
    mo.vstack([sliders, fig_data])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The model, its uncertainty, and where to look next

    We train a small **neural network** to predict the energy from the points we've measured.
    But a single prediction isn't enough — we also need to know **how much to trust it**.

    **Measuring uncertainty with a *deep ensemble*.** Instead of one network, we train
    **four** identical networks that differ only in their random starting weights. Where
    they have data they all agree; where they don't, they each extrapolate differently. That
    **disagreement** (the spread of their predictions) is our uncertainty estimate $\hat\sigma$.

    **The acquisition function — where to measure next.** We want points that are either
    *predicted to be low* (worth exploiting) **or** *uncertain* (worth exploring). The
    **Lower Confidence Bound** captures both with one knob:

    $$ a(x) = \underbrace{\hat\mu(x)}_{\text{predicted energy}} - \;\kappa\;
       \underbrace{\hat\sigma(x)}_{\text{uncertainty}} $$

    We measure next at the point with the **smallest** $a(x)$. Setting $\kappa = 0$ is pure
    greed (always chase the lowest prediction); large $\kappa$ is pure curiosity (always
    chase the unknown). The loop then repeats:

    1. **Train** the ensemble on the data so far.
    2. **Predict** the energy $\hat\mu$ and the uncertainty $\hat\sigma$ everywhere.
    3. **Acquire**: measure the true energy at the point that minimizes $a(x)$.
    4. Add it to the data and go back to step 1.
    """)
    return


@app.cell
def _(ConvergenceWarning, DOMAIN_MAX, MLPRegressor, np, warnings):
    def fit_ensemble(x_train, f_train, n_members, width, y_mean, y_std):
        """Train `n_members` MLPs that differ only in their random init weights."""
        nets = []
        for member in range(n_members):
            net = MLPRegressor(
                hidden_layer_sizes=(width, width),
                solver="lbfgs",          # fast & accurate on small datasets
                alpha=1e-5,
                max_iter=500,
                random_state=member,     # the only thing that differs between members
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.fit(x_train / DOMAIN_MAX, (f_train - y_mean) / y_std)
            nets.append(net)
        return nets

    def ensemble_stats(nets, coords, y_mean, y_std):
        """Return (mean prediction, disagreement) of the ensemble at `coords`."""
        preds = np.stack([net.predict(coords / DOMAIN_MAX) for net in nets])
        mean = preds.mean(axis=0) * y_std + y_mean
        std = preds.std(axis=0) * y_std
        return mean, std

    return ensemble_stats, fit_ensemble


@app.cell
def _(FOUND_TOL, WELL_RADIUS, acquisition, ensemble_stats, fit_ensemble, np):
    def run_bayes_opt(
        x0, y0, funnel, center, n_iter, kappa, target=None,
        # Model size / grid resolution: the levers that set how long a run takes.
        # Tuned so a 15-measurement run is a few seconds natively; the browser (WASM)
        # build is several times slower, so shrink these further if a room is on it.
        n_members=4, width=32, model_n=35,
    ):
        """Active-learning loop: chase the global minimum with an acquisition function.

        Returns per-iteration snapshots so the notebook can replay the search.
        """
        # `center` sets the domain; `target` is the point we are hunting, which is the
        # domain centre on the main landscape but somewhere else on the deceptive one.
        target_x, target_y = target if target is not None else (center, center)
        axis = np.linspace(0.0, 2.0 * center, model_n)
        gx, gy = np.meshgrid(axis, axis)
        coords = np.column_stack([gx.ravel(), gy.ravel()])
        true_vals = funnel(coords[:, 0], coords[:, 1])
        true_min = float(true_vals.min())
        # Mask only the sampled node itself. Anything larger also masks its neighbours,
        # and once a neighbour of the centre is measured the true optimum becomes
        # unreachable — which would make 'find the minimum' unwinnable by construction.
        min_dist = 0.5 * (axis[1] - axis[0])

        x_train = np.column_stack([x0, y0]).astype(float)
        f_train = funnel(x_train[:, 0], x_train[:, 1])
        # Standardisation is fixed on the *initial* data and deliberately not recomputed as
        # points are appended: refitting it mid-run would shift the surrogate's target scale
        # from one iteration to the next and make the snapshots incomparable.
        y_mean = float(f_train.mean())
        y_std = float(f_train.std()) or 1.0

        snapshots = []
        found = None              # first iteration whose best measurement is essentially optimal
        for it in range(n_iter + 1):
            nets = fit_ensemble(x_train, f_train, n_members, width, y_mean, y_std)
            mean, std = ensemble_stats(nets, coords, y_mean, y_std)
            acq = acquisition(mean, std, kappa, it, n_iter)

            # Don't resample where we already have data.
            acq_masked = acq.copy()
            for s in x_train:
                near = np.hypot(coords[:, 0] - s[0], coords[:, 1] - s[1]) < min_dist
                acq_masked[near] = np.inf
            next_pt = coords[np.argmin(acq_masked)]
            # Show the *masked* surface: otherwise the ★ (chosen from the masked one) can
            # sit away from the visibly darkest pixel and the picture contradicts itself.
            acq_shown = np.where(np.isinf(acq_masked), np.nan, acq_masked)

            best_idx = int(np.argmin(f_train))
            best_pt = x_train[best_idx]
            best_energy = float(f_train[best_idx])
            dist = float(np.hypot(best_pt[0] - target_x, best_pt[1] - target_y))
            if found is None and dist < WELL_RADIUS and best_energy - true_min < FOUND_TOL:
                found = it

            snapshots.append(
                {
                    "it": it,
                    "train": x_train.copy(),
                    "mean": mean.reshape(model_n, model_n),
                    "std": std.reshape(model_n, model_n),
                    "err": np.abs(mean - true_vals).reshape(model_n, model_n),
                    "acq": acq_shown.reshape(model_n, model_n),
                    "next": next_pt.copy(),
                    "best_pt": best_pt.copy(),
                    "best_energy": best_energy,
                    "dist": dist,
                    "regret": best_energy - true_min,
                    "n_train": len(x_train),
                    "found": found,
                }
            )

            if it == n_iter:
                break        # the budget is spent: the last ★ is a suggestion, not a measurement
            f_new = funnel(next_pt[0], next_pt[1])
            x_train = np.vstack([x_train, next_pt])
            f_train = np.concatenate([f_train, [f_new]])

        return snapshots, axis, true_vals.reshape(model_n, model_n), true_min

    return (run_bayes_opt,)


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            ### 🎯 Task 2 — What does the model actually know?  *(~6 min)*

            1. Leave the sliders in section 2 at their defaults (25 points, corner 0.25,
               seed 0) and press **▶ Run optimization** below with κ = 1.5 and budget 20.
               It takes a few seconds (longer in the browser version).
            2. When it finishes, drag the **"new measurements taken" slider** (the wide one
               just under the Run button) **back to 0**. That's the model trained on the
               corner data *only*, before it has taken a single new measurement. Study the
               five maps.
            3. Then drag that same slider slowly to the end and watch the search move.

            Answer **Q2.1–Q2.4** on the worksheet: where is the model most uncertain, why do
            the ensemble members disagree there, is uncertainty the same thing as error, and
            when does the search leave the corner?
            """),
            kind="info",
        ),
        mo.accordion({
            "💡 Hint (Task 2)": mo.md(
                "For Q2.3, look at the ripples: near the corner the model has seen the "
                "bumps and knows them; far away it has *smoothed them out*. Does the error "
                "map show mistakes in places the uncertainty map calls 'safe'?"
            ),
        }),
    ])
    return


@app.cell
def _(mo):
    kappa_slider = mo.ui.slider(
        start=0.0, stop=4.0, value=1.5, step=0.5,
        label="κ  —  exploration vs exploitation (0 = pure greed)",
    )
    bo_iters_slider = mo.ui.slider(
        start=5, stop=40, value=20, step=1, label="optimization iterations (budget)"
    )
    bo_run_button = mo.ui.run_button(label="▶ Run optimization")

    bo_controls = mo.vstack(
        [
            mo.md("**Optimization settings** — press Run after changing these."),
            kappa_slider,
            bo_iters_slider,
            bo_run_button,
        ]
    )
    bo_controls
    return bo_iters_slider, bo_run_button, kappa_slider


@app.cell
def _(
    CENTER,
    bo_iters_slider,
    bo_run_button,
    funnel,
    kappa_slider,
    mo,
    run_bayes_opt,
    train_x,
    train_y,
):
    mo.stop(
        not bo_run_button.value,
        mo.md("👆 Set κ and the budget, then press **▶ Run optimization**."),
    )

    bo_snaps, bo_axis, funnel_grid, _funnel_min = run_bayes_opt(
        train_x,
        train_y,
        funnel,
        CENTER,
        n_iter=bo_iters_slider.value,
        kappa=kappa_slider.value,
    )
    return bo_axis, bo_snaps, funnel_grid


@app.cell
def _(bo_snaps, mo):
    bo_iter_slider = mo.ui.slider(
        start=0, stop=len(bo_snaps) - 1, value=0, step=1,
        label="new measurements taken", full_width=True, show_value=True,
    )
    return (bo_iter_slider,)


@app.cell
def _(CENTER, bo_axis, bo_iter_slider, bo_snaps, funnel_grid, go, mo, tidy):
    bo_snap = bo_snaps[min(bo_iter_slider.value, len(bo_snaps) - 1)]

    def opt_markers(show_next=False, show_best=False):
        traces = [
            go.Scatter(
                x=bo_snap["train"][:, 0], y=bo_snap["train"][:, 1], mode="markers",
                marker=dict(color="white", size=4, line=dict(color="black", width=0.5)),
            ),
            go.Scatter(  # the true global optimum
                x=[CENTER], y=[CENTER], mode="markers",
                marker=dict(color="lime", size=13, symbol="diamond",
                            line=dict(color="black", width=1)),
            ),
        ]
        if show_best:
            traces.append(go.Scatter(
                x=[bo_snap["best_pt"][0]], y=[bo_snap["best_pt"][1]], mode="markers",
                marker=dict(color="cyan", size=11, symbol="circle-open",
                            line=dict(color="cyan", width=3)),
            ))
        if show_next:
            traces.append(go.Scatter(
                x=[bo_snap["next"][0]], y=[bo_snap["next"][1]], mode="markers",
                marker=dict(color="red", size=14, symbol="star",
                            line=dict(color="white", width=1)),
            ))
        return traces

    def opt_panel(z, title, colorscale, markers, reverse=False):
        fig = go.Figure(go.Heatmap(
            x=bo_axis, y=bo_axis, z=z, colorscale=colorscale, reversescale=reverse,
            colorbar=dict(thickness=12),
        ))
        for tr in markers:
            fig.add_trace(tr)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            width=350, height=360, showlegend=False,
            xaxis_title="x", yaxis_title="y",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return tidy(fig, left=46, right=70, top=46, bottom=44)

    # Row 1 — the story: what's true, what the model believes, where it wants to go next.
    fig_true_funnel = opt_panel(
        funnel_grid, "① True landscape", "Viridis",
        opt_markers(show_best=True), reverse=True,
    )
    fig_pred_funnel = opt_panel(
        bo_snap["mean"], "② Model's belief  μ̂", "Viridis",
        opt_markers(), reverse=True,
    )
    fig_acq = opt_panel(
        bo_snap["acq"], "③ Where to look next  a(x)", "Cividis",
        opt_markers(show_next=True), reverse=True,
    )
    # Row 2 — the diagnostic pair: what the model *thinks* it doesn't know vs. how wrong
    # it actually is. (In real life you only ever get to see the left one.)
    fig_std = opt_panel(
        bo_snap["std"], "④ Model's uncertainty  σ̂", "Magma", opt_markers(),
    )
    fig_err = opt_panel(
        bo_snap["err"], "⑤ Actual error  |μ̂ − truth|", "Magma", opt_markers(),
    )

    found_txt = (
        f"found the global well after **{bo_snap['found']} new measurements**"
        if bo_snap["found"] is not None
        else "**has not found** the global well yet"
    )

    mo.vstack([
        bo_iter_slider,
        mo.md(
            f"**{bo_snap['n_train']} measurements used** · best energy so far "
            f"**{bo_snap['best_energy']:.2f}** · distance to true optimum "
            f"**{bo_snap['dist']:.1f}** · {found_txt}"
        ),
        mo.md(
            "🟢 true optimum &nbsp;·&nbsp; ◯ best-so-far &nbsp;·&nbsp; ★ next measurement "
            "(on the last frame the budget is gone, so it is only a suggestion) "
            "&nbsp;·&nbsp; white dots = measurements taken<br>"
            "*The small holes in ③ are points we have already measured — the loop blanks them "
            "out so it never spends a measurement twice, which is why the ★ always sits at "
            "the lowest remaining point.*"
        ),
        mo.hstack([fig_true_funnel, fig_pred_funnel, fig_acq],
                  justify="start", widths="equal"),
        mo.md("**Model diagnostics** — ④ is what the model *thinks* it doesn't know; "
              "⑤ is how wrong it actually is. Only ④ is available in real life."),
        mo.hstack([fig_std, fig_err], justify="start"),
    ])
    return


@app.cell
def _(bo_iter_slider, bo_snaps, go, make_subplots, tidy):
    bo_iters = [s["it"] for s in bo_snaps]
    best_e = [s["best_energy"] for s in bo_snaps]
    dist = [s["dist"] for s in bo_snaps]
    cur = min(bo_iter_slider.value, len(bo_snaps) - 1)
    upto = slice(0, cur + 1)  # reveal the curves only up to the current iteration

    # Fixed axis ranges so the curves "draw in" as you scrub, without the plot rescaling.
    e_pad = (max(best_e) - min(best_e)) * 0.08 or 1.0
    d_max = max(dist) or 1.0

    fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_conv.add_trace(
        go.Scatter(x=bo_iters[upto], y=best_e[upto], name="best energy found",
                   mode="lines+markers", line=dict(color="#1f77b4")),
        secondary_y=False,
    )
    fig_conv.add_trace(
        go.Scatter(x=bo_iters[upto], y=dist[upto], name="distance to true optimum",
                   mode="lines+markers", line=dict(color="#2ca02c", dash="dot")),
        secondary_y=True,
    )
    # Emphasize the current iteration (matches the maps above).
    fig_conv.add_trace(
        go.Scatter(x=[bo_iters[cur]], y=[best_e[cur]], mode="markers",
                   marker=dict(color="#1f77b4", size=12, line=dict(color="white", width=1)),
                   showlegend=False),
        secondary_y=False,
    )
    fig_conv.update_xaxes(title_text="new measurements taken",
                          range=[-0.5, max(bo_iters) + 0.5])
    fig_conv.update_yaxes(title_text="best energy found",
                          range=[min(best_e) - e_pad, max(best_e) + e_pad],
                          secondary_y=False)
    fig_conv.update_yaxes(title_text="distance to optimum",
                          range=[0, d_max * 1.08], secondary_y=True)
    fig_conv.update_layout(title="Closing in on the global minimum", width=720, height=400)
    tidy(fig_conv, legend=True, right=95)   # room for the second y-axis title
    fig_conv
    return


@app.cell
def _(mo):
    mo.accordion({
        "🔍 Reveal — Task 2 (open only after you've written your answers)": mo.md(r"""
        - **Uncertainty (④) is highest far from the data** — the top-right half of the
          domain, exactly where nothing has been measured. All four members fit the corner
          data closely (though not identically), but they extrapolate differently, so their
          predictions fan out with distance from the data. Measured on this run, σ̂ tracks
          *distance from the nearest measurement* with a correlation of about **0.9**. That
          fanning-out *is* the uncertainty estimate; there is no probability theory hiding
          behind it.
        - **Uncertainty is not error — σ̂ knows where you haven't looked, not how wrong you
          are.** Compare ④ and ⑤: their bright regions are in different places, and across
          the map the two are barely correlated (measured here: **0.0 – 0.6** depending on
          the seed). Worse, σ̂ is systematically *too small*: on **60–90% of the map the
          real error is larger than σ̂**. The members agree on a smooth, rippleless surface —
          and the truth has ripples none of them has seen, so they agree far more than they
          should. Read σ̂ as "here be dragons", never as an error bar you can trust.
        - **The search leaves the corner almost immediately** (usually within the first few
          iterations): both terms of $a(x)$ point the same way — the model has learned the
          bowl tilts down toward the middle, *and* the middle is unexplored.
        - **It never maps the landscape.** The model's belief (②) stays vague everywhere
          that doesn't matter. Spending measurements there would be wasted budget. That
          selectivity is the entire point of active learning.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The exploration dial, κ

    κ is the one knob that decides **how much curiosity you buy**. Rather than trusting
    anyone's intuition about the right setting, let's measure it: the button below runs the
    *same* loop three times — κ = 0, 1.5 and 4 — from **identical starting data** and a
    budget of 15 measurements, and reports what each one found.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            ### 🎯 Task 3 — The exploration dial  *(~8 min)*

            1. Press **▶ Compare κ values** below (takes ~10 s, longer in the browser) and
               copy the table into
               **Q3.1** on your worksheet.
            2. Look at the three maps of where each run actually measured. Answer **Q3.2**
               (what does κ = 0 do?) and **Q3.3** (what does κ = 4 cost you?).
            3. Now go back to section 2, change the **random seed**, and press
               **▶ Compare κ values** again. Answer **Q3.4** and **Q3.5** — does the same κ
               still win, and what would it actually take to answer "which κ is best"?
            """),
            kind="info",
        ),
        mo.accordion({
            "💡 Hint (Task 3)": mo.md(
                "Count the white dots in the wide-open areas: a κ = 4 run spends "
                "measurements on places it knows nothing about, whether or not they look "
                "promising. And when you compare two numbers that came from *one* run "
                "each, ask yourself how big the difference is compared to how much the "
                "numbers move when you only change the seed."
            ),
        }),
    ])
    return


@app.cell
def _(mo):
    KAPPA_GRID = (0.0, 1.5, 4.0)
    SWEEP_BUDGET = 15
    kappa_run_button = mo.ui.run_button(label="▶ Compare κ values")
    kappa_run_button
    return KAPPA_GRID, SWEEP_BUDGET, kappa_run_button


@app.cell
def _(
    CENTER,
    KAPPA_GRID,
    SWEEP_BUDGET,
    funnel,
    kappa_run_button,
    mo,
    run_bayes_opt,
    train_x,
    train_y,
):
    mo.stop(
        not kappa_run_button.value,
        mo.md("👆 Press **▶ Compare κ values** to run the three settings."),
    )

    kappa_runs = {}
    for sweep_kappa in KAPPA_GRID:
        sweep_snaps, sweep_axis, sweep_truth, _sweep_min = run_bayes_opt(
            train_x, train_y, funnel, CENTER,
            n_iter=SWEEP_BUDGET, kappa=sweep_kappa,
        )
        kappa_runs[sweep_kappa] = sweep_snaps
    return kappa_runs, sweep_axis, sweep_truth


@app.cell
def _(CENTER, KAPPA_GRID, go, kappa_runs, mo, sweep_axis, sweep_truth, tidy):
    kappa_rows = ["| κ | best energy found | measurements to find the global well |",
                  "|---|---|---|"]
    kappa_maps = []
    for panel_kappa in KAPPA_GRID:
        panel_snaps = kappa_runs[panel_kappa]
        panel_last = panel_snaps[-1]
        panel_found = panel_last["found"]
        kappa_rows.append(
            f"| **{panel_kappa}** | {panel_last['best_energy']:.2f} | "
            + (f"{panel_found}" if panel_found is not None else "never")
            + " |"
        )

        panel_fig = go.Figure(go.Heatmap(
            x=sweep_axis, y=sweep_axis, z=sweep_truth,
            colorscale="Viridis", reversescale=True, showscale=False,
        ))
        panel_fig.add_trace(go.Scatter(
            x=panel_last["train"][:, 0], y=panel_last["train"][:, 1], mode="markers",
            marker=dict(color="white", size=5, line=dict(color="black", width=0.5)),
        ))
        panel_fig.add_trace(go.Scatter(
            x=[CENTER], y=[CENTER], mode="markers",
            marker=dict(color="lime", size=12, symbol="diamond",
                        line=dict(color="black", width=1)),
        ))
        panel_fig.update_layout(
            title=dict(text=f"κ = {panel_kappa} — where it measured", font=dict(size=13)),
            width=340, height=340, showlegend=False,
            xaxis_title="x", yaxis_title="y",
        )
        panel_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        tidy(panel_fig, left=46, right=20, top=46, bottom=44)
        kappa_maps.append(panel_fig)

    kappa_curves = go.Figure()
    for curve_kappa, curve_color in zip(KAPPA_GRID, ["#d62728", "#1f77b4", "#2ca02c"]):
        curve_snaps = kappa_runs[curve_kappa]
        kappa_curves.add_trace(go.Scatter(
            x=[s["it"] for s in curve_snaps],
            y=[s["best_energy"] for s in curve_snaps],
            name=f"κ = {curve_kappa}", mode="lines+markers",
            line=dict(color=curve_color, width=2.5),
        ))
    kappa_curves.update_layout(
        title="Best energy found so far (lower is better)",
        xaxis_title="new measurements taken", yaxis_title="best energy found",
        width=720, height=380,
    )
    tidy(kappa_curves, legend=True)

    mo.vstack([
        mo.md("\n".join(kappa_rows)),
        mo.hstack(kappa_maps, justify="start", widths="equal"),
        kappa_curves,
    ])
    return


@app.cell
def _(mo):
    mo.accordion({
        "🔍 Reveal — Task 3 (open only after you've written your answers)": mo.md(r"""
        - **The textbook story is that κ = 0 gets trapped** — pure greed polishes whatever
          well it stumbled into. On *this* landscape it does not merely survive, it **wins**:
          over 150 random seeds its median score is the exact optimum (−3.00, against −2.66
          for κ = 1.5), it finds the global well in 79% of runs against 71%, and it gets
          there in 6 measurements instead of 11. That is worth understanding. The network smooths over the ripples, so its belief is
          basically "a bowl that tilts toward the middle" — and, crucially, everything it
          has *not* measured looks better than the corner it has. Greedily following that
          optimistic belief walks straight to the centre: the **model's inductive bias did
          the exploring**. Note what this does *not* mean — the smoothing alone isn't the
          trick, and a surrogate that reproduces every ripple faithfully still finds the
          centre here. What would break greed is a *pessimistic* model, one that assumes
          unmeasured ground is bad; then nothing pulls it out of the corner and only κ can.
        - **κ = 4 buys curiosity you may not need.** It scatters measurements into empty
          regions regardless of how promising they look, and with a 15-measurement budget
          that is expensive. It is the safer choice when you distrust the model, and the
          wasteful one when the model is already pointing the right way.
        - **One run per κ proves almost nothing.** Change the seed and the ranking often
          flips. To claim "κ = 1.5 is best" you would need many seeds per κ and a look at
          the *spread*, not the single best number — the same discipline you'd apply to any
          noisy experiment. This is the single most common mistake in published
          optimizer comparisons.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The challenge: beat random search on a budget

    A fair question: maybe the landscape is easy and **any** strategy would find the
    minimum. The honest test is the simplest possible baseline — **random search**, which
    just measures at random locations with no model at all.

    Below, both methods get the **same starting data** and the **same budget of 15
    measurements**. Random search is noisy, so it's averaged over 40 repeats (shaded band =
    10th–90th percentile).
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            ### 🎯 Task 4 — Beat the baseline  *(~7 min)*

            Your budget is **15 measurements**. Your score is the **lowest energy you find**
            — lower is better, and a perfect score is **−3.00** at the exact center.

            1. Tune whatever you like — **κ** (section 3), **number of initial points**,
               **corner size**, **seed** (section 2) — and press **▶ Run challenge** (a few seconds).
               Try a few combinations. Record your best in **Q4.1**; you'll be reporting
               that number to the room.
            2. Answer **Q4.2**: did you beat random search, and by how much?
            3. Press **▶ Check across 3 seeds** to re-run *your* settings on three different
               starting datasets, and answer **Q4.3** — does your score hold up?
            4. Answer **Q4.4**: in one sentence, what does the loop have that random search
               doesn't?
            """),
            kind="info",
        ),
        mo.accordion({
            "💡 Hint (Task 4)": mo.md(
                "Two knobs matter more than they look. Extra *initial* points are free — "
                "the 15 measurements are all spent on points the loop *chooses*, on top of "
                "whatever you start with — but every one of them lands in the corner, so "
                "ask yourself whether more of the same corner tells the model anything new. "
                "And a bigger corner means the model sees more of the bowl before it has "
                "to extrapolate. Also: your score can improve just because you got a lucky "
                "seed — which is exactly what step 3 is for."
            ),
        }),
    ])
    return


@app.cell
def _(np):
    def run_random_search(x0, y0, funnel, domain_max, n_iter, seed):
        """Baseline: measure at uniformly random locations, no model. Track best-so-far."""
        rng = np.random.default_rng(seed)
        energies = list(funnel(np.asarray(x0, float), np.asarray(y0, float)))
        best = [float(min(energies))]          # index 0 = the starting data, 0 new measurements
        for _ in range(n_iter):
            nx = rng.uniform(0.0, domain_max)
            ny = rng.uniform(0.0, domain_max)
            energies.append(float(funnel(nx, ny)))
            best.append(float(min(energies)))
        return np.array(best)

    return (run_random_search,)


@app.cell
def _(mo):
    CHALLENGE_BUDGET = 15
    challenge_button = mo.ui.run_button(label="▶ Run challenge")
    seeds_button = mo.ui.run_button(label="▶ Check across 3 seeds")
    mo.hstack([challenge_button, seeds_button], justify="start")
    return CHALLENGE_BUDGET, challenge_button, seeds_button


@app.cell
def _(
    CENTER,
    CHALLENGE_BUDGET,
    DOMAIN_MAX,
    challenge_button,
    funnel,
    go,
    kappa_slider,
    mo,
    np,
    run_bayes_opt,
    run_random_search,
    tidy,
    train_x,
    train_y,
):
    mo.stop(
        not challenge_button.value,
        mo.md("👆 Set your knobs, then press **▶ Run challenge** (a few seconds)."),
    )

    chal_snaps, _chal_axis, _chal_truth, _chal_min = run_bayes_opt(
        train_x, train_y, funnel, CENTER,
        n_iter=CHALLENGE_BUDGET, kappa=kappa_slider.value,
    )
    chal_best = [s["best_energy"] for s in chal_snaps]
    chal_iters = list(range(len(chal_best)))

    chal_rand = np.stack([
        run_random_search(train_x, train_y, funnel, DOMAIN_MAX, CHALLENGE_BUDGET, seed=s)
        for s in range(40)
    ])
    chal_rand_mean = chal_rand.mean(axis=0)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(  # spread of random search
        x=chal_iters + chal_iters[::-1],
        y=list(np.percentile(chal_rand, 90, axis=0))
          + list(np.percentile(chal_rand, 10, axis=0)[::-1]),
        fill="toself", fillcolor="rgba(150,150,150,0.2)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig_cmp.add_trace(go.Scatter(
        x=chal_iters, y=chal_rand_mean, name="random search (avg of 40)",
        line=dict(color="gray", dash="dot"),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=chal_iters, y=chal_best, name="active learning (your run)",
        line=dict(color="#1f77b4", width=3),
    ))
    fig_cmp.update_layout(
        title="Your run vs. random guessing (lower is better)",
        xaxis_title="new measurements taken", yaxis_title="best energy found",
        width=720, height=420,
    )
    tidy(fig_cmp, legend=True)

    chal_score = chal_best[-1]
    chal_margin = chal_rand_mean[-1] - chal_score
    # With the optimum now reachable, ties at exactly -3.00 are common — so report how
    # quickly it got there as well. That is the leaderboard tie-break.
    chal_found = chal_snaps[-1]["found"]
    chal_speed = (
        f" · found the global well after **{chal_found}** measurements"
        if chal_found is not None else " · never found the global well"
    )

    mo.vstack([
        mo.md(
            f"## 🏆 Your score: **{chal_score:.2f}**\n\n"
            f"κ = **{kappa_slider.value}** · {CHALLENGE_BUDGET} measurements · "
            f"random search averaged **{chal_rand_mean[-1]:.2f}** with the same budget "
            f"(you beat it by **{chal_margin:.2f}**){chal_speed}."
            if chal_margin > 0 else
            f"## 🏆 Your score: **{chal_score:.2f}**\n\n"
            f"κ = **{kappa_slider.value}** · {CHALLENGE_BUDGET} measurements · "
            f"random search averaged **{chal_rand_mean[-1]:.2f}** — it beat you this time "
            f"by **{-chal_margin:.2f}**{chal_speed}."
        ),
        fig_cmp,
    ])
    return


@app.cell
def _(
    CENTER,
    CHALLENGE_BUDGET,
    DOMAIN_MAX,
    DOMAIN_MIN,
    corner_frac_slider,
    funnel,
    kappa_slider,
    mo,
    n_points_slider,
    np,
    run_bayes_opt,
    seed_slider,
    seeds_button,
):
    mo.stop(
        not seeds_button.value,
        mo.md("👆 Press **▶ Check across 3 seeds** to re-run your settings on three "
              "different starting datasets."),
    )

    seed_scores = []
    for check_seed in (seed_slider.value, seed_slider.value + 101, seed_slider.value + 202):
        check_rng = np.random.default_rng(check_seed)
        check_hi = DOMAIN_MIN + corner_frac_slider.value * (DOMAIN_MAX - DOMAIN_MIN)
        check_x = check_rng.uniform(DOMAIN_MIN, check_hi, size=n_points_slider.value)
        check_y = check_rng.uniform(DOMAIN_MIN, check_hi, size=n_points_slider.value)
        check_snaps, _cx, _ct, _cm = run_bayes_opt(
            check_x, check_y, funnel, CENTER,
            n_iter=CHALLENGE_BUDGET, kappa=kappa_slider.value,
        )
        seed_scores.append((check_seed, check_snaps[-1]["best_energy"],
                            check_snaps[-1]["found"]))

    seed_table = ["| starting seed | score (best energy) | measurements to find the global well |",
                  "|---|---|---|"]
    for row_seed, row_score, row_found in seed_scores:
        seed_table.append(
            f"| {row_seed} | {row_score:.2f} | "
            + (f"{row_found}" if row_found is not None else "never") + " |"
        )
    seed_vals = [s[1] for s in seed_scores]
    seed_table.append(
        f"| **spread** | **{min(seed_vals):.2f} … {max(seed_vals):.2f}** | |"
    )

    mo.md("\n".join(seed_table))
    return


@app.cell
def _(mo):
    mo.accordion({
        "🔍 Reveal — Task 4 (open only after you've written your answers)": mo.md(r"""
        - **The active-learning curve drops further — and, at κ = 1.5, it takes most of the
          budget to get there.** Measured on the shipped default: the grey random-search
          average is *ahead of* the loop at 8 of the first 12 points on this plot (after
          three measurements random sits at 1.23 — the on-screen average of 40 repeats; 1.34
          as a median over thousands — against the loop's 1.58). The loop only takes a lead
          it keeps at its **12th** of 15 measurements, and finishes at −0.94 against random's
          −0.40: a margin of about half an energy unit, won late. The loop spends its early
          picks buying information rather than scoring, and cashes it in at the end — on this
          particular run it barely gets to cash in at all. Be careful not to read the shipped
          default as typical, in either direction: it is an **unlucky draw**, worse than 15
          of 20 fresh seeds, whose median is **−2.66** against random's **−0.60**. That
          median is the method's real margin here; this one run is a reminder that a single
          run is not evidence. At κ = 0 the loop's first pick doesn't improve on the
          starting data either, so random's average leads at that first point too — but from
          its **second** measurement onward κ = 0 is ahead and stays ahead, ending at the
          exact optimum.
          Random search meanwhile shows steady diminishing returns — measured on this
          landscape its distance from the optimum falls off roughly like a power law, so each
          further halving of the gap costs you something like 3× more guesses — because a new
          guess only helps if it happens to beat everything before it. The loop instead *uses*
          every measurement twice: once as a candidate answer, and once as information that
          reshapes where it looks next. That's the thing random search doesn't have — a
          **memory**, in the form of a model.
        - **How much is that worth?** To match what the loop typically achieves in 15
          measurements at κ = 1.5, uniform random guessing needs a median of **≈260 guesses**
          (10th–90th percentile 40–840) — about a 17× saving. Matching κ = 0's typical result
          is worse than that: its median *is* the exact optimum, which continuous random
          sampling hits with probability zero.
        - **The gap is believed to grow with difficulty and dimension** — in 20D random
          search is hopeless while model-guided search still works, which is why real
          campaigns (materials, molecules, instrument tuning) bother. Note that this
          notebook does *not* demonstrate that: everything here is 2D. Take it as the
          motivation for the method, not as something you just measured. (And be fair to the
          baseline: our own grid search over the acquisition function would die in 20D too —
          real implementations optimize it instead of gridding it.)
        - **Your score is part skill, part luck.** The 3-seed check usually spreads by more
          than the difference between neighbouring κ values. If the class leaderboard is
          ranked on one run each, the winner is partly the luckiest, not the wisest —
          which is exactly how optimizer benchmarks get published, too.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. One more landscape: what if the trend lies?

    Everything so far lived on a landscape with an **honest** shape: the broad bowl really
    did tilt toward the answer. That is why greed did so well — the model learned the tilt
    from the corner and simply followed it downhill.

    Real problems are not always so kind. Here is the same eggbox, but now the bowl bottoms
    out at a **decoy**: a shallow, ordinary dip that the smooth trend points straight at. The
    genuinely deep spot — the thing you actually want — sits in a **small pocket on the other
    side of the map**, and *nothing* in the broad shape hints that it is there.

    Same loop, same budget, same starting corner. Only the landscape changed.

    *(Two details, so nothing here surprises you: this landscape keeps its ripples fixed, so
    the `roughness` slider from section 1 does not change it; and the two runs below use a
    starting set of their own rather than your sliders, so that everyone sees the same
    comparison.)*
    """)
    return


@app.cell
def _(DOMAIN_MAX, np):
    # A deliberately deceptive twin of the main landscape. The bowl's bottom is a decoy;
    # the true minimum hides in a two-scale basin (a wide, shallow "hint" wrapped around a
    # narrow, deep core) diagonally opposite. This is the classic two-funnel construction
    # used by the BBOB benchmark suite's f24 (Lunacek bi-Rastrigin) to defeat greedy search.
    DEC_AXIS = np.linspace(0.0, DOMAIN_MAX, 35)
    DECOY_XY = (DEC_AXIS[10], DEC_AXIS[24])   # bottom of the broad bowl — but only shallow
    HIDDEN_XY = (DEC_AXIS[24], DEC_AXIS[10])  # the true global minimum

    def deceptive(x, y, roughness=3.0):
        """Eggbox ripples + a bowl centred on a decoy + a hidden two-scale basin."""
        centre = DOMAIN_MAX / 2.0
        ripple = -roughness * np.cos(0.5 * (x - centre)) * np.cos(0.5 * (y - centre))
        bowl = 2.5 * ((x - DECOY_XY[0]) ** 2 + (y - DECOY_XY[1]) ** 2) / centre ** 2
        gap2 = (x - HIDDEN_XY[0]) ** 2 + (y - HIDDEN_XY[1]) ** 2
        hidden = (
            -3.0 * np.exp(-gap2 / (2 * 8.0 ** 2))    # wide, shallow hint
            - 6.0 * np.exp(-gap2 / (2 * 3.0 ** 2))   # narrow, deep core
        )
        return ripple + bowl + hidden

    return DECOY_XY, HIDDEN_XY, deceptive


@app.cell
def _(
    DECOY_XY,
    DOMAIN_MAX,
    DOMAIN_MIN,
    HIDDEN_XY,
    deceptive,
    go,
    mo,
    np,
    tidy,
):
    # The same two views section 1 gives the first landscape, so the two can be compared.
    dec_view_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 160)
    dec_vx, dec_vy = np.meshgrid(dec_view_axis, dec_view_axis)
    dec_view_z = deceptive(dec_vx, dec_vy)

    dec_marks = dict(
        x=[DECOY_XY[0], HIDDEN_XY[0]], y=[DECOY_XY[1], HIDDEN_XY[1]],
        text=["decoy — where the bowl points", "the real minimum"],
    )

    dec_surface = go.Figure(go.Surface(
        x=dec_view_axis, y=dec_view_axis, z=dec_view_z,
        colorscale="Viridis", reversescale=True,
        colorbar=dict(title="energy", thickness=12),
    ))
    # Stems rather than bare markers: the real minimum sits at the bottom of a pocket, so a
    # marker placed on the surface is hidden by the surface itself from most camera angles.
    # (3D text labels collide too — the top-down tab carries the labelling.)
    for dec_mx, dec_my, dec_colour in zip(dec_marks["x"], dec_marks["y"],
                                          ["orange", "lime"]):
        dec_surface.add_trace(go.Scatter3d(
            x=[dec_mx, dec_mx], y=[dec_my, dec_my],
            z=[float(deceptive(dec_mx, dec_my)), float(dec_view_z.max())],
            mode="lines+markers",
            line=dict(color=dec_colour, width=5),
            marker=dict(color=dec_colour, size=6, line=dict(color="black", width=1)),
        ))
    dec_surface.update_layout(
        title="The deceptive landscape — 🟠 decoy, 🟢 the real minimum",
        width=680, height=480, showlegend=False,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="energy"),
    )
    tidy(dec_surface, left=0, right=60, bottom=0)

    dec_map = go.Figure(go.Heatmap(
        x=dec_view_axis, y=dec_view_axis, z=dec_view_z,
        colorscale="Viridis", reversescale=True,
        colorbar=dict(title="energy", thickness=12),
    ))
    dec_map.add_trace(go.Scatter(
        x=[DECOY_XY[0]], y=[DECOY_XY[1]], mode="markers+text", text=["decoy"],
        textposition="top center", textfont=dict(color="orange"),
        marker=dict(color="orange", size=13, symbol="x", line=dict(width=2)),
    ))
    dec_map.add_trace(go.Scatter(
        x=[HIDDEN_XY[0]], y=[HIDDEN_XY[1]], mode="markers+text", text=["the real minimum"],
        textposition="top center", textfont=dict(color="lime"),
        marker=dict(color="lime", size=13, symbol="diamond",
                    line=dict(color="black", width=1)),
    ))
    dec_map.add_shape(
        type="rect", x0=DOMAIN_MIN, y0=DOMAIN_MIN,
        x1=0.25 * DOMAIN_MAX, y1=0.25 * DOMAIN_MAX,
        line=dict(color="white", width=1.5, dash="dash"),
    )
    dec_map.update_layout(
        title="Top-down — dark = deep · dashed box = where the measurements start",
        width=560, height=500, showlegend=False,
        xaxis_title="x", yaxis_title="y",
    )
    dec_map.update_yaxes(scaleanchor="x", scaleratio=1)
    tidy(dec_map, right=90)

    mo.vstack([
        mo.ui.tabs({"top-down map": dec_map, "3D surface": dec_surface}),
        mo.md(
            "The broad bowl of this landscape is centred on the **✕** — so from the dashed "
            "starting box, the only trend a smooth model can pick up points that way. The "
            "**🟢** is far deeper, but it sits in a small pocket, and nothing in the shape "
            "of the surface near the start hints that it is there."
        ),
    ])
    return



@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(r"""
            ### 🎯 Task 5 — When the trend lies  *(~5 min)*

            Press **▶ Run both on the deceptive landscape** below. It runs the loop twice —
            once fully greedy (κ = 0) and once curious (κ = 4) — from the **same starting
            data**, so the only thing that differs is the dial.

            Answer **Q5.1–Q5.3** on the worksheet: where does each run spend its
            measurements, which one finds the hidden pocket, and what had to be true about
            the *first* landscape for greed to look so good there.
            """),
            kind="info",
        ),
        mo.accordion({
            "💡 Hint (Task 5)": mo.md(
                "Watch where the white dots pile up. Greed has no reason to leave a place "
                "that is already the best thing it has seen — and on this landscape the "
                "best thing it has seen is a trap."
            ),
        }),
    ])
    return


@app.cell
def _(mo):
    deceptive_button = mo.ui.run_button(label="▶ Run both on the deceptive landscape")
    deceptive_button
    return (deceptive_button,)


@app.cell
def _(
    CENTER,
    DOMAIN_MAX,
    HIDDEN_XY,
    deceptive,
    deceptive_button,
    mo,
    np,
    run_bayes_opt,
):
    mo.stop(
        not deceptive_button.value,
        mo.md("👆 Press **▶ Run both on the deceptive landscape** (~10 s)."),
    )

    # This comparison uses a FIXED starting set of its own, not the sliders above, so that
    # everyone in the room sees the same two runs and the only difference between them is κ.
    # (Any single pair of runs is still one sample — the Reveal box gives the spread over 24
    # random starts, which is what you should actually believe.)
    dec_rng = np.random.default_rng(7)
    dec_hi = 0.25 * DOMAIN_MAX
    dec_x0 = dec_rng.uniform(0.0, dec_hi, 25)
    dec_y0 = dec_rng.uniform(0.0, dec_hi, 25)

    dec_runs = {}
    for dec_kappa in (0.0, 4.0):
        dec_snaps, dec_axis, dec_truth, dec_min = run_bayes_opt(
            dec_x0, dec_y0, deceptive, CENTER,
            n_iter=15, kappa=dec_kappa, target=HIDDEN_XY,
        )
        dec_runs[dec_kappa] = (dec_snaps, dec_axis, dec_truth, dec_min)
    return (dec_runs,)


@app.cell
def _(DECOY_XY, HIDDEN_XY, dec_runs, go, mo, tidy):
    dec_panels = []
    dec_rows = ["| | κ = 0 (pure greed) | κ = 4 (curious) |", "|---|---|---|"]
    dec_scores = {}
    for dec_k in (0.0, 4.0):
        dec_s, dec_ax, dec_z, dec_true = dec_runs[dec_k]
        dec_last = dec_s[-1]
        dec_scores[dec_k] = (dec_last["best_energy"], dec_last["dist"], dec_true)

        dec_fig = go.Figure(go.Heatmap(
            x=dec_ax, y=dec_ax, z=dec_z, colorscale="Viridis", reversescale=True,
            showscale=False,
        ))
        dec_fig.add_trace(go.Scatter(
            x=dec_last["train"][:, 0], y=dec_last["train"][:, 1], mode="markers",
            marker=dict(color="white", size=5, line=dict(color="black", width=0.5)),
        ))
        dec_fig.add_trace(go.Scatter(
            x=[DECOY_XY[0]], y=[DECOY_XY[1]], mode="markers+text",
            marker=dict(color="orange", size=12, symbol="x", line=dict(width=2)),
            text=["decoy"], textposition="top center", textfont=dict(color="orange"),
        ))
        dec_fig.add_trace(go.Scatter(
            x=[HIDDEN_XY[0]], y=[HIDDEN_XY[1]], mode="markers+text",
            marker=dict(color="lime", size=13, symbol="diamond",
                        line=dict(color="black", width=1)),
            text=["the real one"], textposition="top center", textfont=dict(color="lime"),
        ))
        dec_fig.update_layout(
            title=dict(text=f"κ = {dec_k} — where it measured", font=dict(size=13)),
            width=360, height=360, showlegend=False,
            xaxis_title="x", yaxis_title="y",
        )
        dec_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        tidy(dec_fig, left=46, right=20, top=46, bottom=44)
        dec_panels.append(dec_fig)

    dec_rows.append(
        f"| best energy found | **{dec_scores[0.0][0]:.2f}** | "
        f"**{dec_scores[4.0][0]:.2f}** |"
    )
    dec_rows.append(
        f"| how close it got to the pocket | **{dec_scores[0.0][1]:.1f}** away | "
        f"**{dec_scores[4.0][1]:.1f}** away |"
    )

    mo.vstack([
        mo.md(
            f"The deepest point on this landscape is **{dec_scores[0.0][2]:.2f}**, in the "
            "pocket marked 🟢. The bowl's own bottom — the ✕ — only reaches about −3."
        ),
        mo.hstack(dec_panels, justify="start", widths="equal"),
        mo.md("\n".join(dec_rows)),
    ])
    return


@app.cell
def _(mo):
    mo.accordion({
        "🔍 Reveal — Task 5 (open only after you've written your answers)": mo.md(r"""
        - **Greed goes to the decoy and stays there.** It has no reason to leave the best
          thing it has seen, and the smooth trend keeps confirming that choice. Measured over
          24 random starting sets on this landscape: κ = 0 ends up with a median score of
          **−0.5**, and gets into the hidden pocket **0 times out of 24**. κ = 4 scores a
          median of **−3.7** — more than three energy units better — against a true minimum
          of −8.58, and κ = 4 beat κ = 0 on **20 of 24** starting sets.
          Actually bottoming out the narrow core is rare for either (1 of 24 even for κ = 4):
          the win is in getting to the right region at all.
        - **So what made greed look so good on the first landscape?** That the trend was
          *honest*. The bowl really did tilt toward the answer, the network learned the tilt
          from a corner of data, and following it blindly was a winning strategy. Greed
          wasn't clever there — it was **lucky in the structure of the problem**. This is the
          real lesson: the right amount of exploration is a property of the landscape you are
          on, not a universal constant, and you usually cannot tell which landscape you are
          on until afterwards.
        - **An honest caveat about this second landscape**: here the loop does *not*
          convincingly beat random search (measured medians −3.7 for κ = 4 against −2.9 for
          random — a gap that does not clear the noise at the sample sizes we ran). Finding a small pocket that nothing points to is
          genuinely hard, and 15 measurements in two dimensions is not much. The first
          landscape is where active learning proves its worth; this one is where exploration
          proves it is necessary. Neither landscape makes both points at once, and pretending
          otherwise would be the kind of overclaiming this notebook is trying to teach you to
          distrust.
        """),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Wrap-up — questions worth arguing about

    - Each measurement here takes microseconds, and we fit four neural networks between
      measurements. **When is that a terrible trade? When is it obviously worth it?**
    - What if every measurement came back **noisy** — the same point giving a different
      answer each time? What breaks?
    - This landscape has 2 inputs. Real problems often have 20. What happens to random
      search? What happens to the grid our model searches over?
    - We told the loop to find the *minimum*. What would change if you wanted an accurate
      model **everywhere** instead?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 🧑‍💻 Optional: change the rule the loop follows

    Everything above uses one line of code, in the cell below: the **acquisition function**
    that scores every candidate point. Edit it and re-run the challenge (section 5) to see
    what happens. Suggestions are in the worksheet:

    - `return -std` — pure curiosity, ignore the prediction entirely.
    - `return mean` — pure greed (same as κ = 0).
    - `return mean - kappa * (1 - it / max(n_iter, 1)) * std` — a **cooling schedule**:
      bold early, greedy later.
    """)
    return


@app.cell
def _():
    # ──────────────────────────────────────────────────────────────────────────────
    #  👇 THE ACQUISITION RULE — this is the one line the whole loop turns on.
    #     `mean` = the ensemble's prediction μ̂, `std` = its disagreement σ̂,
    #     `it` = current iteration, `n_iter` = total budget. Lower score = measure here.
    # ──────────────────────────────────────────────────────────────────────────────
    def acquisition(mean, std, kappa, it, n_iter):
        """Lower Confidence Bound: trade predicted energy against uncertainty."""
        return mean - kappa * std

    return (acquisition,)


if __name__ == "__main__":
    app.run()
