# Finding a needle in a rough landscape 🥚🔎

### Student worksheet — active learning / Bayesian optimization

**Time:** ~30 minutes on your own, then ~10 minutes discussing together.
**You need:** this page and the notebook —
[ibengtsson.github.io/eggbox](https://ibengtsson.github.io/eggbox/), which runs in your
browser with nothing to install (give it a minute to boot).
**You do not need to write any code** — everything is sliders and buttons. (There is an
optional code task at the end if you finish early.)
**Keep a pen and paper (or a scratch file) handy** for your answers.
**Be patient with the ▶ Run buttons:** each one retrains the model from scratch, which takes
a few seconds — quite a bit longer in the zero-install browser version. Read the next
question while you wait.

---

## The setup

You can *measure* some quantity at any point `(x, y)` — the energy of a material, the yield
of a reaction, the score of a design — but **each measurement is expensive**: a lab
experiment, a long simulation. Somewhere in that landscape is a **global minimum** and you
want to find it. You have a budget of a few dozen measurements. Where do you spend them?

**Active learning** answers: fit a quick model to what you have measured so far, let it tell
you where it is worth measuring next, measure there, refit, repeat. When the goal is
specifically to find the optimum, this loop is called **Bayesian optimization**.

Your job today is to find out *whether that actually works*, and *what makes it work or
fail*. Jot your answers down as you go — we will pool them across the room at the end, and
the disagreements between your answers are the interesting part.

---

## Task 1 — Why is this hard? *(~4 min)*

Open **section 1** of the notebook, "The landscape: a rough funnel".

1. Drag the **roughness** slider from `0` to `4` and back. Watch the surface.
2. Leave it at `3` (the default) before moving on.

**Q1.1** With roughness at 0 the surface is a single smooth bowl. Roughly **how many separate
dips (local minima)** can you count when roughness is 3?

**Q1.2** Imagine you start at the bottom-left corner and simply **walk downhill** (this is
what gradient descent does). Where do you end up, and why is that a problem?

**Q1.3** Now imagine you have no model at all and just **guess at random**. Using your count
from Q1.1: roughly what is the chance that **one** random guess lands in the *correct* dip —
the deep one at the centre?

---

## Task 2 — What does the model actually know? *(~6 min)*

Scroll to **section 2**. All of our measurements so far sit in the **bottom-left corner** —
nowhere near the centre. Leave the sliders at their defaults (25 points, corner size 0.25,
seed 0).

Now go to **section 3** and press **▶ Run optimization** once with the default settings
(κ = 1.5, budget 20). It takes a few seconds — noticeably longer if you're on the
browser version. When it finishes, drag the **"new measurements taken" slider** (the wide
one just under the Run button) **back to 0** — that shows you the model trained on *only* the
corner data, before it has taken a single new measurement.

You are looking at five maps: the **true landscape**, the model's **prediction**, **where it
wants to measure next**, the model's **uncertainty**, and its **error** (how wrong the
prediction actually is).

**Q2.1** Where is the model's **uncertainty** highest? Describe the region in one sentence.

**Q2.2** The model's uncertainty comes from **ensemble disagreement**: four identical
networks are trained on the same data and differ only in their random starting weights.
Why do they agree in the corner but disagree far away from it?

**Q2.3** Compare the **uncertainty** map with the **error** map. Are they the same picture?
Where do they differ, and what does that tell you about trusting an uncertainty estimate?

**Q2.4** Now drag that same slider slowly to the end and watch the measurements appear.
At roughly which iteration does the search **leave the corner**? Does it map the whole
landscape on the way, or not?

---

## Task 3 — The exploration dial, κ *(~8 min)*

The loop picks the next measurement at the smallest value of

$$a(x) = \hat\mu(x) - \kappa\,\hat\sigma(x)$$

— predicted energy *minus* κ times uncertainty. **κ = 0** is pure greed: always go where the
model predicts the lowest value. **Large κ** is pure curiosity: always go where the model
knows least.

In **section 4**, press **▶ Compare κ values**. This runs the same loop three times, at
κ = 0, 1.5 and 4, from identical starting data, and fills in the table for you.

**Q3.1** Note down all three rows of the table it produces — the κ value, the best energy it
found, and how many measurements it took to find the global well.

**Q3.2** Look at the κ = 0 run's map. What goes wrong (or right)? Describe its behaviour in
one sentence.

**Q3.3** And κ = 4? Why might *too much* curiosity also be a bad deal when measurements are
expensive?

**Q3.4** Now change the **random seed** (section 2) to a different value and press
**▶ Compare κ values** again. Does the *same* κ win?

**Q3.5** So: from your two runs, can you honestly say which κ is "best"? What would you have
to do to answer that properly?

---

## Task 4 — The challenge: beat random search on a budget *(~7 min)*

Scroll to **section 5**. You now have a hard budget of **15 measurements** on top of your
starting data. Your score is the **lowest energy you find** — smaller (more negative) is
better. The perfect score is **−3.00**, at the exact centre.

You may tune: **κ**, the **number of initial points**, the **corner size**, and the **seed**.
Press **▶ Run challenge** to score an attempt (a few seconds each). Try a few.

**Q4.1** Note down your best attempt: the κ, number of initial points, corner size and seed
you used, and the **score** (best energy) it got. Bring that score to the discussion — and
also note **how many measurements it took to find the global well**, which the notebook
reports next to your score. A perfect −3.00 is reachable, so ties are likely and that second
number is the tie-break.

**Q4.2** The plot compares your run against **random search** given the same budget (grey
band = the spread over 1000 random repeats). Did you beat it, and by how much in energy? Then
estimate: judging by how slowly the grey curve is falling, would *doubling* random search's
budget have been enough to catch you?

**Q4.3** Press **▶ Check across 3 seeds**. Your winning settings are re-run on three
different starting datasets. Does your score hold up?

**Q4.4** One sentence: **why** does the active-learning loop beat random search here? Name
the thing it has that random search does not.

---

## Task 5 — When the trend lies *(~5 min)*

Everything so far happened on a landscape with an **honest** shape: the broad bowl really did
tilt toward the answer, which is why greed did so well. Section 6 shows the same eggbox with
one change — the bowl now bottoms out at a **decoy**, and the genuinely deep spot hides in a
small pocket elsewhere that no smooth trend points at.

Press **▶ Run both on the deceptive landscape**. It runs the loop twice, at κ = 0 and κ = 4,
from the *same* starting data — so the dial is the only difference.

**Q5.1** Look at where each run spent its measurements. Describe the difference in one
sentence each.

**Q5.2** Which run found the hidden pocket, and what score did each end up with?

**Q5.3** Back to Task 3 for a moment: κ = 0 did *well* on the first landscape and badly here.
What was true about the first landscape that made greed look so good — and could you have
known that in advance, on a real problem?

---

## Wrap-up questions — think about these before we discuss

- Each measurement here is a function call taking microseconds, and we fit four neural
  networks between measurements. **When is that a terrible trade?** When is it obviously
  worth it?
- What if every measurement came back with **noise** — the same point giving a different
  answer each time? What would break?
- This landscape has 2 inputs. Real problems often have 20. What happens to random search
  in 20 dimensions? What happens to the grid the model searches over?
- We told the loop to find the *minimum*. What would change if you instead wanted an
  accurate model **everywhere**?

---

## Finished early? Optional code task 🧑‍💻

This one needs a notebook where the code is visible: use the
[**`/edit/` version**](https://ibengtsson.github.io/eggbox/edit/) (or run it locally with
`uv run --with marimo marimo edit eggbox_active_learning.py --sandbox`). At the bottom
there is one clearly marked cell containing the acquisition rule:

```python
def acquisition(mean, std, kappa, it, n_iter):
    return mean - kappa * std      # Lower Confidence Bound
```

Change that one `return` and rerun the challenge:

1. **Pure curiosity:** `return -std` — ignore the prediction entirely. Does it still find the
   minimum? Is it efficient?
2. **Pure greed:** `return mean` — same as κ = 0.
3. **Cooling schedule:** start bold, get greedy — shrink κ as the loop proceeds, using the
   iteration number `it`: `return mean - kappa * (1 - it / max(n_iter, 1)) * std`.
   Does it beat a fixed κ?

Report what you find — this is a real research question, not a solved one.
