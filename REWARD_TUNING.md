# Dodo Stand-and-Balance — Reward Tuning & Debugging Guide

How to read the training logs, decide whether the policy is actually learning,
and tune the `Dodo-Stand-v0` rewards when it misbehaves (wobbles, drifts, falls,
or freezes). Pairs with `source/dodo_rl/dodo_rl/tasks/locomotion/stand_env_cfg.py`.

---

## 1. The one rule: watch episode length, not mean reward

`mean reward` is the **summed return over an episode**, so it goes up just by
running longer episodes — it is NOT a clean measure of "is it standing." The
honest signal is **mean episode length** climbing toward the cap (it's surviving)
and **`Episode_Termination/time_out` dominating `base_contact`** (it's timing out
upright, not falling). Don't lengthen `episode_length_s` to chase a bigger reward
number — that's a metric illusion.

Also: with pushes + mass/COM randomization + obs noise ON in training, reward is
**lower and noisier** than a clean run. That's expected, not failure.

---

## 2. Where the per-term breakdown lives

Every reward term is logged separately, in two places:

- **Console**, each iteration: lines like `Episode_Reward/alive`,
  `Episode_Reward/flat_orientation_l2`, plus `Episode_Termination/base_contact`
  and `Episode_Termination/time_out`.
- **TensorBoard**: `tensorboard --logdir logs/rsl_rl/dodo_stand` →
  groups `Episode_Reward/*`, `Episode_Termination/*`, and `Train/mean_reward`,
  `Train/mean_episode_length`. The curves are much easier than the console.

Each `Episode_Reward/<term>` is the **per-episode sum, already weighted**, averaged
across envs — so magnitudes are directly comparable between terms.

---

## 3. What healthy looks like

- Positive terms rising and staying positive: `alive`, `track_lin_vel_xy_exp`,
  `track_ang_vel_z_exp`.
- Penalties small and drifting toward 0: `flat_orientation_l2`, `lin_vel_z_l2`,
  `ang_vel_xy_l2`, `feet_slide`, `joint_deviation_all`, `action_rate_l2`.
- `termination_penalty` (−200 on a fall) climbing toward 0.
- `Episode_Termination/time_out` >> `base_contact`.
- Balance converges fast — episode length should be clearly rising within
  ~200–300 iterations. If not, **debug, don't add iterations**.

---

## 4. Reward terms — meaning, sign, and what to do

| Term | Weight (stand) | Pulls toward | Raise it to… | Lower it to… |
|------|----------------|--------------|--------------|--------------|
| `alive` | +1.0 | surviving each step | survive longer (rarely needed) | — |
| `track_lin_vel_xy_exp` | +1.0, `std=0.25` | zero base translation (hold still) | punish drift harder (tighten `std`) | allow more drift |
| `track_ang_vel_z_exp` | +1.0 | zero yaw rate | stop spinning | — |
| `flat_orientation_l2` | −2.0 | body level | fight tipping | if too rigid |
| `lin_vel_z_l2` | −2.0 | no vertical bob | stop bouncing | — |
| `ang_vel_xy_l2` | −0.25 | no roll/pitch rate | **kill wobble/rocking** | if sluggish under push |
| `feet_slide` | −0.5 | feet planted | **stop walking/sliding** | if it can't reposition feet to recover |
| `feet_air_time` | 0.0 | (off) | — | keep 0 (any + value rewards stepping) |
| `action_rate_l2` | −0.05 | smooth actions | **de-twitch / de-wobble** | if it reacts too slowly to a push |
| `dof_acc_l2` | −2.5e-7 | low joint accel | damp joint jitter | if motion too damped |
| `joint_deviation_all` | −0.2 | home pose | hold pose tighter | **if it freezes and can't catch a push** |
| `termination_penalty` | −200 | not falling | (leave) | — |

---

## 5. Triage flow

1. **Termination split first.** `base_contact` >> `time_out` and short episodes →
   it's **falling**. Don't tune individual rewards yet:
   - confirm the printed joint order matches `POLICY_JOINT_ORDER` (rl_env.py),
   - check pushes aren't simply too strong (turn `push_robot.params.velocity_range`
     down to ±0.1, or off, and see if it can stand at all),
   - confirm action scale is 0.5 (pinned in `rough_env_cfg.py`).
2. **Surviving (`time_out` dominates) but reward low / behavior bad** → find the
   single **largest-magnitude penalty** and **halve its weight**. Retrain a few
   hundred iters. Recheck.
3. **Change ONE weight at a time** so you can attribute the effect.

---

## 6. Symptom → fix recipes

**Wobbling / jittering (underdamped).** Most physical cause is the low joint
damping (`kd=0.5`, kept low to match the real motors), so we damp via reward
instead. Raise `ang_vel_xy_l2` (−0.1 → −0.25), `action_rate_l2` (−0.01 → −0.05),
and `dof_acc_l2` (×2). *(Applied in the current cfg.)* If still buzzy, push
`action_rate_l2` to −0.1.

**Walking / drifting (won't hold a spot).** Raise `feet_slide` (−0.2 → −0.5) and
tighten `track_lin_vel_xy_exp` `std` (0.5 → 0.25). *(Applied.)* If it still steps,
go `feet_slide` −0.8 and `std` 0.2. Avoid any positive `feet_air_time`.

**Tips over slowly / can't catch a push.** It's too stiff/passive. **Lower**
`joint_deviation_all` (−0.2 → −0.1 or −0.05) and `action_rate_l2` (back to −0.02)
so it's allowed to move to recover. Make sure pushes are actually on in training.

**Freezes / goes limp and falls.** A penalty dominates the positive signal — check
which `Episode_Reward/*` penalty is largest and halve it (usually `action_rate_l2`,
`dof_torques_l2`, or `joint_deviation_all`).

**Won't learn at all (flat reward, short episodes from step 0).** Setup bug, not
reward — joint order, action scale, obs layout, or pushes too strong. See §5.1.

---

## 7. Sanity checks baked into the scripts

`scripts/rsl_rl/train.py` and `play.py` print, right after env creation:

```python
print(env.unwrapped.scene["robot"].joint_names)
print(env.unwrapped.cfg.decimation, env.unwrapped.cfg.sim.dt, env.unwrapped.step_dt)
```

Expect `decimation=4, sim.dt=0.005, step_dt=0.02` (the 50 Hz contract), and the
joint order MUST equal `POLICY_JOINT_ORDER` in the hardware `rl_env.py` or the
deployed policy drives the wrong joints (looks like instant flailing). Delete
these prints once confirmed.

---

## 8. PLAY vs TRAIN

- `Dodo-Stand-v0` (train): pushes + randomization ON.
- `Dodo-Stand-Play-v0` (play): pushes OFF, noise OFF — a clean "can it stand?"
  demo. To eyeball **push resistance**, comment out `self.events.push_robot = None`
  in `DodoStandEnvCfg_PLAY` and the trained-in pushes come back.

Usual flow: Play quiet to confirm it stands still cleanly → re-enable pushes in
Play to confirm it takes a shove and recovers.
