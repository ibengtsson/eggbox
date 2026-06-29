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

    This notebook is a gentle, hands-on introduction to **active learning** for people who
    are new to machine learning.

    Imagine you can *measure* some quantity at any point `(x, y)` — the energy of a material,
    the yield of a reaction, the score of a design — but **each measurement is expensive**
    (a lab experiment, a long simulation). The landscape is **rough**: full of dips and
    bumps, with one **global minimum** hiding somewhere that you want to find. Checking every
    point on a fine grid would take thousands of measurements. Can we be smarter?

    **Active learning** says yes: train a quick model on what you've measured so far, let it
    tell you **where it's worth measuring next**, measure there, and repeat. Done well, you
    can pin down the global minimum in a few dozen measurements instead of thousands. When
    the goal is specifically *finding the optimum*, this loop is also called **Bayesian
    optimization**.

    Our playground is the **"eggbox"** — a bumpy, periodic surface from the MultiNest paper
    ([arXiv:0809.3437](https://arxiv.org/pdf/0809.3437)) — tilted into a broad bowl so that
    one well, at the center, is the deepest. We'll start stuck in a corner and watch the loop
    hunt the center down.
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
    from sklearn.neural_network import MLPRegressor
    from plotly.subplots import make_subplots
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    return ConvergenceWarning, MLPRegressor, make_subplots, warnings


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
    global minimum** (marked 🟢). The `roughness` slider sets how deep the local wells are:
    at 0 it's a single smooth bowl (trivial); turn it up and the traps get deeper.
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

    def funnel(x, y):
        """The true landscape. Low = good; the global minimum is the central well."""
        ripple = -roughness_slider.value * (
            np.cos(0.5 * (x - CENTER)) * np.cos(0.5 * (y - CENTER))
        )
        r2 = (x - CENTER) ** 2 + (y - CENTER) ** 2
        bowl = BOWL_STRENGTH * r2 / BOWL_RADIUS ** 2
        return ripple + bowl

    return CENTER, DOMAIN_MAX, DOMAIN_MIN, funnel


@app.cell
def _(CENTER, DOMAIN_MAX, DOMAIN_MIN, funnel, go, mo, np):
    # Preview of the landscape (updates live with the roughness slider).
    funnel_axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 160)
    fxx, fyy = np.meshgrid(funnel_axis, funnel_axis)
    funnel_z = funnel(fxx, fyy)
    z_opt = float(funnel(CENTER, CENTER))

    fig_funnel_top = go.Figure(
        go.Heatmap(
            x=funnel_axis, y=funnel_axis, z=funnel_z,
            colorscale="Viridis", reversescale=True,
            colorbar=dict(title="energy", thickness=12),
        )
    )
    fig_funnel_top.add_trace(
        go.Scatter(
            x=[CENTER], y=[CENTER], mode="markers",
            marker=dict(color="lime", size=13, symbol="diamond",
                        line=dict(color="black", width=1)),
        )
    )
    fig_funnel_top.update_layout(
        title="Rough funnel (top-down)",
        xaxis_title="x", yaxis_title="y",
        width=480, height=480, showlegend=False,
        margin=dict(l=40, r=10, t=40, b=40),
    )
    fig_funnel_top.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_funnel_surface = go.Figure(
        go.Surface(
            x=funnel_axis[::2], y=funnel_axis[::2], z=funnel_z[::2, ::2],
            colorscale="Viridis", reversescale=True, showscale=False,
        )
    )
    fig_funnel_surface.add_trace(
        go.Scatter3d(
            x=[CENTER], y=[CENTER], z=[z_opt], mode="markers",
            marker=dict(color="lime", size=6, symbol="diamond",
                        line=dict(color="black", width=1)),
        )
    )
    fig_funnel_surface.update_layout(
        title="The same funnel in 3D (🟢 = global minimum)",
        width=480, height=480, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="energy",
                   aspectmode="cube"),
    )

    mo.hstack([fig_funnel_top, fig_funnel_surface], justify="start", widths="equal")
    return funnel_axis, funnel_z


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Start with data in one corner

    Suppose all we've measured so far sits in the **bottom-left corner** — maybe that's where
    we happened to start looking. The global minimum is far away at the center, and we have
    no data near it. The sliders set how much of the corner we sampled and how many points
    we have.
    """)
    return


@app.cell
def _(mo):
    n_points_slider = mo.ui.slider(
        start=5, stop=100, value=25, step=5, label="number of initial points"
    )
    corner_frac_slider = mo.ui.slider(
        start=0.1, stop=0.5, value=0.25, step=0.05,
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
        width=560, height=520, showlegend=False,
    )
    fig_data.update_yaxes(scaleanchor="x", scaleratio=1)

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
    **several** identical networks that differ only in their random starting weights. Where
    they have data they all agree; where they don't, they each extrapolate differently. That
    **disagreement** (the spread of their predictions) is our uncertainty estimate.

    **The acquisition function — where to measure next.** We want points that are either
    *predicted to be low* (worth exploiting) **or** *uncertain* (worth exploring). The
    **Lower Confidence Bound** captures both with one knob:

    $$ a(x) = \underbrace{\hat\mu(x)}_{\text{predicted energy}} - \;\kappa\;
       \underbrace{\hat\sigma(x)}_{\text{uncertainty}} $$

    We measure next at the point with the **smallest** $a(x)$. Setting $\kappa = 0$ is pure
    greed (always chase the lowest prediction); large $\kappa$ is pure curiosity (always
    chase the unknown). The loop then repeats:

    1. **Train** the ensemble on the data so far.
    2. **Predict** the energy and the uncertainty everywhere.
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
def _(ensemble_stats, fit_ensemble, np):
    def run_bayes_opt(x0, y0, funnel, center, n_members, width, n_iter, kappa, model_n):
        """Bayesian optimization: chase the minimum with an LCB acquisition function."""
        axis = np.linspace(0.0, 2.0 * center, model_n)
        gx, gy = np.meshgrid(axis, axis)
        coords = np.column_stack([gx.ravel(), gy.ravel()])
        true_vals = funnel(coords[:, 0], coords[:, 1])
        true_min = float(true_vals.min())
        min_dist = (axis[-1] - axis[0]) / model_n * 1.5

        x_train = np.column_stack([x0, y0]).astype(float)
        f_train = funnel(x_train[:, 0], x_train[:, 1])
        y_mean = float(f_train.mean())
        y_std = float(f_train.std()) or 1.0

        snapshots = []
        for it in range(n_iter + 1):
            nets = fit_ensemble(x_train, f_train, n_members, width, y_mean, y_std)
            mean, std = ensemble_stats(nets, coords, y_mean, y_std)
            acq = mean - kappa * std            # Lower Confidence Bound (we minimize)

            # Don't resample where we already have data.
            acq_masked = acq.copy()
            for s in x_train:
                near = np.hypot(coords[:, 0] - s[0], coords[:, 1] - s[1]) < min_dist
                acq_masked[near] = np.inf
            next_pt = coords[np.argmin(acq_masked)]

            best_idx = int(np.argmin(f_train))
            best_pt = x_train[best_idx]
            best_energy = float(f_train[best_idx])

            snapshots.append(
                {
                    "it": it,
                    "train": x_train.copy(),
                    "mean": mean.reshape(model_n, model_n),
                    "acq": acq.reshape(model_n, model_n),
                    "next": next_pt.copy(),
                    "best_pt": best_pt.copy(),
                    "best_energy": best_energy,
                    "dist": float(np.hypot(best_pt[0] - center, best_pt[1] - center)),
                    "regret": best_energy - true_min,
                    "n_train": len(x_train),
                }
            )

            f_new = funnel(next_pt[0], next_pt[1])
            x_train = np.vstack([x_train, next_pt])
            f_train = np.concatenate([f_train, [f_new]])

        return snapshots, axis, true_vals.reshape(model_n, model_n), true_min

    return (run_bayes_opt,)


@app.cell
def _(mo):
    kappa_slider = mo.ui.slider(
        start=0.0, stop=4.0, value=1.5, step=0.5,
        label="κ  —  exploration vs exploitation (0 = pure greed)",
    )
    bo_iters_slider = mo.ui.slider(
        start=5, stop=40, value=30, step=1, label="optimization iterations (budget)"
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

    bo_snaps, bo_axis, funnel_grid, funnel_min = run_bayes_opt(
        train_x,
        train_y,
        funnel,
        CENTER,
        n_members=5,
        width=48,
        n_iter=bo_iters_slider.value,
        kappa=kappa_slider.value,
        model_n=45,
    )
    return bo_axis, bo_snaps, funnel_grid


@app.cell
def _(bo_snaps, mo):
    bo_iter_slider = mo.ui.slider(
        start=0, stop=len(bo_snaps) - 1, value=0, step=1,
        label="optimization iteration", full_width=True, show_value=True,
    )
    return (bo_iter_slider,)


@app.cell
def _(CENTER, bo_axis, bo_iter_slider, bo_snaps, funnel_grid, go, mo):
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
            title=title, width=360, height=380, showlegend=False,
            margin=dict(l=40, r=10, t=40, b=40), xaxis_title="x", yaxis_title="y",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # True landscape (reversed so the deep global well is bright), prediction, acquisition.
    fig_true_funnel = opt_panel(
        funnel_grid, f"True landscape (iter {bo_snap['it']})", "Viridis",
        opt_markers(show_best=True), reverse=True,
    )
    fig_pred_funnel = opt_panel(
        bo_snap["mean"], f"Model's belief (iter {bo_snap['it']})", "Viridis",
        opt_markers(), reverse=True,
    )
    fig_acq = opt_panel(
        bo_snap["acq"], f"Where to look next (iter {bo_snap['it']})", "Cividis",
        opt_markers(show_next=True), reverse=True,
    )

    mo.vstack([
        bo_iter_slider,
        mo.md(
            f"**{bo_snap['n_train']} evaluations** · best energy so far "
            f"**{bo_snap['best_energy']:.2f}** · distance to true optimum "
            f"**{bo_snap['dist']:.1f}** &nbsp;·&nbsp; 🟢 true optimum, "
            f"◯ best-so-far, ★ next."
        ),
        mo.hstack([fig_true_funnel, fig_pred_funnel, fig_acq],
                  justify="start", widths="equal"),
    ])
    return


@app.cell
def _(bo_iter_slider, bo_snaps, go, make_subplots):
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
    fig_conv.update_xaxes(title_text="optimization iteration",
                          range=[-0.5, max(bo_iters) + 0.5])
    fig_conv.update_yaxes(title_text="best energy found",
                          range=[min(best_e) - e_pad, max(best_e) + e_pad],
                          secondary_y=False)
    fig_conv.update_yaxes(title_text="distance to optimum",
                          range=[0, d_max * 1.08], secondary_y=True)
    fig_conv.update_layout(
        title="Closing in on the global minimum",
        width=720, height=360, legend=dict(orientation="h", y=1.15),
    )
    fig_conv
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What to notice

    - The loop starts trapped in the corner but quickly **stops refining what it already
      knows and strikes out toward the unexplored center**, where the acquisition function
      (right) promises lower energy. Within ~15–20 evaluations the best-so-far point (◯)
      snaps onto the true global optimum (🟢).
    - It does this **without mapping the landscape** — the model's belief (middle) stays
      vague in regions that don't matter. Spending precious measurements there would be
      wasted budget. That selectivity is the whole point.
    - **Try κ = 0** (pure greed): the loop often gets stuck polishing a local well it
      stumbled into, because it never gambles on the unknown. Crank **κ up** and it explores
      more widely — sometimes finding the center faster, sometimes wasting budget wandering.
      That tension is the central trade-off of optimization under uncertainty.
    - Turn **roughness** down to 0 and the problem becomes a single smooth bowl — trivial.
      Turn it up and the local wells get deep enough to swallow a greedy search.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Is this better than just guessing?

    A fair question: maybe the landscape is easy and **any** strategy would find the
    minimum. The honest test is to compare against the simplest possible baseline —
    **random search**, which just measures at random locations with no model at all.

    Below we give both methods the **same starting data and the same budget**, and track the
    **best energy found so far**. Random search is noisy, so we average it over many repeats
    (shaded band = 10th–90th percentile). The active-learning curve is the single run you
    watched above.
    """)
    return


@app.cell
def _(np):
    def run_random_search(x0, y0, funnel, domain_max, n_iter, seed):
        """Baseline: measure at uniformly random locations, no model. Track best-so-far."""
        rng = np.random.default_rng(seed)
        energies = list(funnel(np.asarray(x0, float), np.asarray(y0, float)))
        best = []
        for _ in range(n_iter + 1):
            best.append(float(min(energies)))
            nx = rng.uniform(0.0, domain_max)
            ny = rng.uniform(0.0, domain_max)
            energies.append(float(funnel(nx, ny)))
        return np.array(best)

    return (run_random_search,)


@app.cell
def _(DOMAIN_MAX, bo_snaps, funnel, go, np, run_random_search, train_x, train_y):
    n_iter_cmp = len(bo_snaps) - 1
    iters_cmp = list(range(n_iter_cmp + 1))
    bo_best = [s["best_energy"] for s in bo_snaps]

    rand = np.stack([
        run_random_search(train_x, train_y, funnel, DOMAIN_MAX, n_iter_cmp, seed=s)
        for s in range(40)
    ])
    rand_mean = rand.mean(axis=0)
    rand_lo = np.percentile(rand, 10, axis=0)
    rand_hi = np.percentile(rand, 90, axis=0)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(  # spread of random search
        x=iters_cmp + iters_cmp[::-1],
        y=list(rand_hi) + list(rand_lo[::-1]),
        fill="toself", fillcolor="rgba(150,150,150,0.2)",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig_cmp.add_trace(go.Scatter(
        x=iters_cmp, y=rand_mean, name="random search (avg of 40)",
        line=dict(color="gray", dash="dot"),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=iters_cmp, y=bo_best, name="active learning (LCB)",
        line=dict(color="#1f77b4", width=3),
    ))
    fig_cmp.update_layout(
        title="Active learning vs. random guessing (lower is better)",
        xaxis_title="measurements taken", yaxis_title="best energy found",
        width=720, height=380, legend=dict(orientation="h", y=1.15),
    )
    fig_cmp
    return


@app.cell
def _(mo):
    mo.md(r"""
    The active-learning curve drops **faster and further**: it reaches a deep energy in a
    handful of measurements, while random search is still stumbling around. With a tight
    measurement budget — the situation that matters when each point is a real experiment —
    that gap is the difference between finding the answer and running out of money.

    *(On an easy landscape — roughness near 0, or a tiny domain — the two can tie: if
    guessing is good enough, you don't need to be clever. The advantage grows with the
    difficulty and dimensionality of the problem.)*
    """)
    return


if __name__ == "__main__":
    app.run()
