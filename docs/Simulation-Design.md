**1. Timing & discretization**

Three separate rates, decided now so nothing downstream is ambiguous:
- **Physics substep**: 0.002s (500Hz) — MuJoCo's default integrator, fine enough to integrate the pendulum accurately even near the unstable equilibrium. If you ever see energy drift in a free-swing test, switch to `integrator="RK4"` before shrinking the timestep further.
- **Decision rate**: 20Hz (0.05s) — this is the rate you log frames/actions at, *and* the rate your data-collection controller (swing-up/LQR) recomputes its control signal. That's 25 physics substeps per decision.
- **Render rate**: irrelevant for data collection if you're using `mujoco.Renderer` for offscreen capture — you only need to render at the decision rate, not 60Hz, since you're not watching it live.

The important design rule, and the actual fix for the `ts=0.02s` vs `20Hz` mismatch in your diagram: **the controller itself must only update at the decision rate, holding its output constant across all 25 substeps until the next decision.** If instead the LQR ran at the full 500Hz internally and you merely sampled one snapshot of it every 25th step to log, the logged action would just be an approximate summary of what was actually applied — and your predictor is trained to assume `a_t` was the *exact, constant* control over the whole interval from `o_t` to `o_{t+1}`. Get this wrong and you've quietly broken the Markov assumption the whole predictor relies on.

**2. MuJoCo model**

Slider joint (cart, `range` set to your track half-length, `limited="true"`) + hinge joint (pole) + motor actuator on the slider with a bounded `ctrlrange`. Give the slider a hard joint limit rather than handling bounds in Python — this makes wall-contact data appear naturally in your dataset (cart pins or bounces at the rail) without any special termination logic, and it's a physically real behavior for the predictor to learn.

**3. Data-collection controller: swing-up + LQR**

Energy-based swing-up (Åström–Furuta), with θ=0 at bottom, θ=π upright:
- Pendulum energy: E(θ, θ̇) = ½ m l² θ̇² + m g l (1 − cos θ)
- Target energy at top: E_top = 2 m g l
- Control: `a_cmd = k_swing · (E − E_top) · sign(θ̇ · cos θ)`, saturated to your actuator's force limit.
- Watch for this in practice: the pure energy law says nothing about cart position, so it can walk the cart toward one rail while pumping energy. If you see that in testing, add a small proportional term pulling the cart back toward center.

Switch to LQR once `|θ − π| < θ_sw` (start with ~20°) and `|θ̇| < ω_sw` (a couple rad/s), with a wider hysteresis band (~35°) before switching back — otherwise you'll get chattering right at the boundary.

LQR: linearize about upright (deviation state `s = [x, ẋ, θ−π, θ̇]`), get `(A,B)` either from the standard textbook cart-pole linearization or numerically from your exact MuJoCo model via finite differences at a fine step size, solve the continuous algebraic Riccati equation for `K`, apply `u = −K·s`. Compute this once per decision step using the true state at that instant — same "hold constant across substeps" rule as above. I can derive the exact linearized matrices for your specific mass/length choices, or sketch the finite-difference approach against MuJoCo directly, if useful once you're at that step.

**4. Randomization & noise**

Per episode, randomize:
- **θ₀**: uniform over the full range (don't bias toward bottom — near-vertical starts are cheap "free" coverage of the region you most need data in).
- **x₀**: uniform within ~60% of the track half-length (avoid always starting flush against a wall).
- **k_swing**: ±30–50% around a hand-tuned nominal.
- **θ_sw**: something like 15°–35°.
- **LQR aggressiveness**: scale the whole `Q` matrix by a log-uniform factor (e.g. [0.5, 2]×) rather than randomizing each entry independently — preserves relative weighting and is much less likely to hand you a non-stabilizing gain by accident. Check closed-loop eigenvalues stay stable across your sampled range; if some draws are marginally unstable, that's fine *as long as it's intentional* (gives you recovery-dynamics data).
- **Action noise**: start simple — fixed-std Gaussian added to the commanded force each decision step, something like 2–5% of max actuator force. Refine later (e.g. reduce it once switched to LQR) only if you find you're not getting enough quiet, stable, near-vertical frames.

Optional, worth deciding now rather than retrofitting later: reserve ~10–15% of episodes with noticeably wider noise/parameter randomization as a "stress" subset. Doesn't cost much and hedges against the predictor being poorly calibrated on states your main controller family never really visits.

**5. Episode structure**

~10 seconds per episode (200 steps at 20Hz) — long enough to contain a full swing-up (typically 1–3s) plus a meaningful balanced tail. No need for separate goal-image collection: after gathering data, scan episodes for stretches where `|θ−π|` and `|θ̇|` both stay small for, say, ≥1 second, and cache a few of those frames as goal-image candidates. This only works if your noise scheme reliably produces quiet balanced stretches — another reason to sanity-check that before scaling up.

**6. Storage**

Recommend a single HDF5 file, one group per episode: `frames` (uint8, `[T,H,W,3]`), `actions` (float32, `[T,1]`), `qpos_qvel` (float32, `[T,4]`) for ground truth (used only for validation/probing, never for training), plus per-episode attrs recording the sampled controller parameters and initial conditions — you'll want that for debugging and for reproducing any particular episode later. HDF5 gives you cheap random access across episodes, which matters: SIGReg needs batches drawn from *many different states across the dataset*, not just one contiguous trajectory, so your loader should be built for that access pattern from the start. Split train/val **by episode**, not by frame.

**7. Build order — validate each piece before combining**

1. MJCF + free-swing sanity check (drop the pole from a few angles, eyeball that it looks physically right).
2. LQR alone, zero noise: start near-upright, confirm it balances indefinitely.
3. Swing-up alone, zero noise: start at bottom, confirm it reaches the switch band reliably, confirm the switch to LQR doesn't chatter.
4. Turn on randomization + noise across a batch of episodes (start with ~50), plot `θ(t)` for a handful, and check the aggregate histogram of visited `θ` values — you want real mass both near the bottom/swinging region and near the top.
5. Only once 1–4 look right, run the full collection.

**8. Starting size**

Given this is a 1-DoF-cart + 1-DoF-pole system (much lower-dimensional than the paper's manipulation tasks), I'd start well below their 10–20k episodes — something like **500–1000 episodes** (100k–200k frame-action pairs total) is a reasonable first pass. See how coverage and training look before deciding whether to scale up.
