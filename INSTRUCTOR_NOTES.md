# Instructor notes — eggbox active-learning session

Companion to `WORKSHEET.md` and `eggbox_active_learning.py`. Not for distribution before
the session (it contains the answers).

## Shape of the session (~35 min)

| | | |
|---|---|---|
| 0:00 | **Set up & frame** — hand out the worksheet, give the notebook link, one sentence on the premise: *expensive measurements, find the minimum.* Do **not** pre-explain κ. | 3 min |
| 0:03 | **Tasks 1–4**, students work alone or in pairs. Float and look for the failure modes below. | ~25 min |
| 0:28 | **Wrap-up discussion** (script below). Collect leaderboard scores first — it takes 30 seconds and buys you the best discussion of the session. | ~8 min |

Pairs work better than solo: the arguments they have while filling in Q2.3 and Q3.5 are the
learning. Tell them **not** to open the 🔍 Reveal boxes until their answer is written.

**Timing reality check** (measured on a laptop, native): a 15-measurement run is ~3.5 s, so
Task 2's budget-20 run is ~5 s, Task 3's κ comparison ~11 s per press (two presses), and
Task 4 ~3.5 s per attempt plus ~11 s for the three-seed check. That's ~1 minute of waiting
per student. **On the WASM/GitHub Pages build everything is roughly 8× slower** — the same
sequence is closer to 5–6 minutes of waiting, plus ~1 minute for the page itself to boot.
Plan for that: see "Logistics" below.

## Expected answers

### Task 1 — Why is this hard?

- **Q1.1** The ripples repeat every $4\pi$; the domain is $10\pi$ wide, so the dips sit on a
  5×5 grid of stationary points of which **13 are wells**. Anything in the "about a dozen"
  range is right. The exact number matters less than the realisation that there are *many*.
- **Q1.2** Downhill from the corner lands in **the nearest local well and stops** — a
  perfectly good answer to the wrong question. Gradient descent has no mechanism for leaving
  a well; nothing in it knows that a deeper one exists elsewhere.
- **Q1.3** About **1 in 13 ≈ 8 %** per guess (by area of the central cell it's ~16 %; either
  reasoning is fine). Push on the follow-up: *and how many guesses to be confident?* — this
  is the number Task 4 makes them measure.

*Common wrong turn:* students count the bright bumps rather than the dark wells and get ~12
either way, which is fine. Someone will say "infinitely many, it's continuous" — a good
answer; ask them to count *basins* instead.

### Task 2 — What does the model know?

- **Q2.1** Uncertainty is highest **far from the data**, i.e. everything away from the
  bottom-left corner, growing with distance.
- **Q2.2** All four networks fit the corner data equally well, so they agree there. Away
  from data, nothing in the loss constrains them; each one's random initialisation sends it
  off in a different direction. The spread is a *proxy* for "I have no evidence here" — no
  probability theory involved.
- **Q2.3** **They are not the same picture.** The shapes are similar, but there are places
  where the ensemble is confidently wrong: the members agree on a smooth, ripple-free
  surface far from the corner, so σ̂ is modest while the true landscape has deep ripples
  there. This is *the* point of the task — an uncertainty estimate is a model output, not a
  guarantee, and deep ensembles are known to be over-confident under distribution shift.
- **Q2.4** It leaves the corner **within the first handful of iterations** (often iteration
  1–3), and it **does not map the landscape** — large parts of the domain are never
  measured. Selectivity is the whole point.

### Task 3 — The κ dial

**Numbers vary by seed and by run — that is the lesson, not a bug.** Measured over 6 seeds
with the shipped settings (25 corner points, 15-measurement budget; the true minimum is
−3.00, and "found" means within 0.5 of it):

| κ | best energy over 6 seeds | measurements to find the global well | median |
|---|---|---|---|
| 0 | −2.36 … −3.00 | 5, 6, 9, 10, 11, never | **−2.66** |
| 1.5 | −1.02 … −3.00 | 4, 8, 11, 14, never ×2 | **−2.66** |
| 4 | +0.31 … −3.00 | 9, 11, 12, never ×3 | **−2.51** |

Read down the columns before the session: **the spread within a κ is as large as the
difference between κ values.** That is the whole of Q3.5.

- **Q3.2** The textbook expectation is that **κ = 0 gets trapped** polishing one well. Here
  it usually does *fine*, and this is the most interesting thing in the whole notebook: the
  neural network **smooths over the ripples**, so its belief is essentially "a bowl tilting
  toward the middle" — and greedily following that belief walks straight to the centre. The
  **model's inductive bias did the exploring.** Ask the room: *what would have to change for
  greed to fail?* (A model that interpolates the ripples faithfully — a GP with a short
  length scale, say — or a landscape whose global structure isn't learnable from a corner.)
- **Q3.3** κ = 4 spends measurements on empty regions regardless of how promising they look.
  With 15 measurements, that is expensive; it is the right insurance only when you distrust
  the model.
- **Q3.4 / Q3.5** The ranking often **flips with the seed**. Honest answer: from one run per
  κ you cannot say anything. You'd need many seeds per κ and a look at the *spread* — the
  same discipline as any noisy experiment. Optimizer papers get this wrong all the time.

*Common wrong turn:* "κ = 1.5 won so κ = 1.5 is best." Ask what their error bar is.

### Task 4 — Beat the baseline

- **Q4.1** Scores cluster around **−2.5 to −3.0** for good settings; **−3.00** is perfect.
  Note the shipped defaults (κ = 1.5, seed 0, 25 points) are *not* tuned to look good — a
  default run often lands near −1, which is the point: there is something to beat.
- **Q4.2** Random search with 15 measurements averages ≈ **−0.6** (10th–90th percentile
  −2.3 … +0.6 over 200 repeats), against ≈ −2.7 for a typical loop run. Random
  search improves like the *log* of the number of guesses; ask how many guesses it would
  need to match their 15 (usually hundreds — off the right edge of the plot).
- **Q4.3** Scores usually spread by **1 energy unit or more** across three seeds — often more
  than the difference between adjacent κ values in Task 3. This is the setup for the
  leaderboard reveal.
- **Q4.4** The loop has a **model — a memory**. Every measurement is used twice: as a
  candidate answer *and* as information that changes where it looks next. Random search
  throws the information away.

## Wrap-up script (~8 min)

1. **Collect the leaderboard** (1 min). Read out scores, write the top 3 on the board with
   their settings. Ask the winner what they tuned.
2. **"Who here got a different best κ than their neighbour?"** (2 min). Hands will go up.
   Land the point: single-run comparisons of stochastic optimizers are noise. Show the
   3-seed spread from Q4.3 next to the leaderboard gaps — usually the spread is bigger than
   the gaps, i.e. **the leaderboard partly ranks luck**. (Say plainly that this is how a lot
   of published benchmark tables are produced.)
3. **"Did anyone's κ = 0 run beat their κ = 1.5 run?"** (2 min). Most hands. Then the good
   bit: *why doesn't greed fail here?* → the surrogate's smoothing is doing the exploration
   (Q3.2 above). Generalise: **your uncertainty estimate and your inductive bias are part of
   the algorithm**, not neutral machinery.
4. **Uncertainty ≠ error** (1 min). Call back to Q2.3. In real campaigns you only ever see
   the σ̂ map, and it is confidently wrong exactly where you have no data to notice.
5. **Where this actually pays** (2 min). Microsecond measurements + four neural nets = a
   terrible trade; a week-long synthesis or a 10 000-core simulation = obvious. Then
   dimension: random search dies in 20D, and so does our grid-based `argmin` over the
   acquisition function — real implementations optimize the acquisition function instead of
   gridding it. Close on the loop being the spine of self-driving labs.

If time runs short, drop item 4 — it is the one they can read in the Reveal box.

## Logistics

- **Native is ~8× faster than the browser build** (measured: a 15-measurement challenge run
  takes ~3.5 s natively and around half a minute in WASM, after a ~1 minute page boot). If
  you can, have students run locally:
  `uv run --with marimo marimo edit eggbox_active_learning.py --sandbox`. The GitHub Pages
  (WASM) build needs no install but runs everything in Pyodide — expect runs to take
  noticeably longer. If the room is on WASM, tell them to press a Run button and read the
  next task while it works, and consider cutting Task 3's second seed.
- **The optional code task needs an editable notebook** — the deployed app-mode page hides
  the code. Point fast finishers at the `/edit/` version of the deployed page, or at a local
  `marimo edit` session.
- Everything is reactive: changing the seed or roughness slider invalidates the last run, and
  the plots are replaced by "press Run". That is expected — tell them so up front, or you
  will answer it fifteen times.
- Nothing is saved. If a student wants their numbers, they write them on the sheet.

## Knobs you might want to change before the session

- `BOWL_STRENGTH = 6.0` (in the funnel cell) sets how strongly the bowl tilts toward the
  centre. Lowering it makes greed fail more often and the problem harder — but convergence
  gets much less reliable within 15–30 measurements (validated across seeds: at 3.0 the loop
  frequently never reaches the central well). Only weaken it if you want a harder,
  less predictable session, and re-check across seeds first.
- The κ grid compared in Task 3 is `KAPPA_GRID` in the section-4 cell; the budgets are
  `SWEEP_BUDGET` (15) and `CHALLENGE_BUDGET` (15).
- Ensemble size / width / grid resolution are arguments of `run_bayes_opt` (`n_members=4`,
  `width=32`, `model_n=35`) — the levers for making runs faster on slow machines. These were
  benchmarked against the previous, 2× more expensive setting (5 × 48-wide nets on a 45²
  grid) with no meaningful loss in the outcomes students see.
- **Runs are not exactly reproducible even at a fixed seed**: scikit-learn's `lbfgs` solver
  isn't bitwise deterministic across BLAS thread counts, so the same settings can give a
  different score on a re-press. Worth saying out loud if a student notices — it reinforces
  the Task 3 lesson rather than undermining it.
