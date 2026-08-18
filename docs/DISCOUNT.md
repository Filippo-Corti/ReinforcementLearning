# The Discounted Horizon

This file records what $\gamma$ is actually doing in this project, measured
rather than assumed, and what the reported experiments should use.

The learning contract fixes $\gamma=0.9995$ for all three algorithms. That
choice is uncomfortable for three reasons: the task is finite-horizon, so no
discount is needed to make the return converge; the value is so close to one
that it looks like a rounding of "no discount"; and nothing recorded so far
says what it buys. This is the measurement that answers it.

## What was compared

Thirty runs: three algorithms $\times$ two discounts $\times$ five seeds, each
at the full $2{,}000{,}000$-interaction budget, on the fixed Experiment 1
circuit with the medium actor, the Frenet observation, and the calibrated
learning rates. Everything except $\gamma$ is held at what the reported
experiments will use.

The two runs of a pair share a seed. Pairing matters more here than the sample
size: this task's seed-to-seed spread is large enough that an unpaired
comparison of five runs would mostly measure the seeds. For the same reason the
grid is sharded by *pair*, so both members of a pair run on the same machine at
the same time, and a difference between the two workers cannot arrive disguised
as a difference between the two discounts.

Every end-of-training figure is the mean over the **final quarter** of the forty
checkpoints. Evaluation is one deterministic episode per checkpoint, so a single
final point measures where the jitter landed.

A further six runs form the critic-rate control described in section 6.

## 1. The discount is not load-bearing, and $\gamma=1$ is well-posed

Every episode ends — by crashing, stalling, finishing, or reaching
$T_{\max}=1000$ — so the undiscounted return is a finite sum by construction.
The usual reason for $\gamma<1$, keeping an infinite sum convergent, does not
apply.

The one thing that could have made $\gamma=1$ *wrong* is mishandled truncation,
and it is handled correctly. The workers do not auto-reset: each returns the
true final observation and the engine resets it explicitly afterwards, and
`compute_gae_targets` bootstraps with

$$ \text{bootstrap} = \begin{cases} 0 & \text{terminated} \\ V(s_{t+1}) & \text{otherwise.} \end{cases} $$

So at the step cap A2C and PPO continue the return through $V(s_T)$ rather than
pretending the episode ended.

> REINFORCE is the exception: `monte_carlo_return_to_go` treats truncation as a
> true ending and bootstraps nothing. This is a pre-existing bias at both
> discounts, and $\gamma=1$ removes the decay that softened it. It is
> **empirically irrelevant here**: training episodes reach the step cap
> $0.6\%$ of the time, and that fraction does not differ between the two
> discounts ($0.0061$ against $0.0068$).

## 2. For A2C and PPO the trace, not the discount, sets the horizon

GAE weights a temporal-difference error $k$ steps ahead by $(\gamma\lambda)^k$.
With $\lambda=0.95$:

| | $\gamma=0.9995$ | $\gamma=1$ |
|---|---:|---:|
| $\gamma\lambda$ | 0.949525 | 0.950000 |
| effective span $1/(1-\gamma\lambda)$ | 19.81 steps | 20.00 steps |
| weight 20 steps ahead | 0.3549 | 0.3585 |

The discount changes the actor's credit-assignment span by **one percent**. Both
A2C and PPO also standardize advantages before the actor loss, so the $60\%$
wider advantage spread measured at $\gamma=1$ never reaches the actor's gradient
scale either.

For those two algorithms, therefore, the only channel the discount has left is
the **critic's regression targets**, which are much wider without it:

| | $\gamma=0.9995$ | $\gamma=1$ | pairs |
|---|---:|---:|---:|
| A2C value-target SD | 5.10 | 8.12 | 5/5 |
| PPO value-target SD | 5.87 | 9.51 | 5/5 |
| A2C critic loss | 21.9 | 39.2 | 5/5 |
| PPO critic loss | 20.2 | 44.4 | 5/5 |

See `outputs/discount_credit_weights.png`.

## 3. Removing the discount slows the car — in every algorithm and every seed

Mean speed over the final quarter of training, in metres per second:

| | $\gamma=0.9995$ | $\gamma=1$ | difference | vs seed spread | pairs favouring $\gamma=1$ |
|---|---:|---:|---:|---:|---:|
| REINFORCE | 19.28 | 16.85 | $-2.43\pm1.11$ | 2.8$\times$ | 0/5 |
| A2C | 20.14 | 18.64 | $-1.50\pm0.60$ | 5.5$\times$ | 0/5 |
| PPO | 21.35 | 18.48 | $-2.87\pm1.48$ | 3.1$\times$ | 0/5 |

And the lap times that follow from it, in seconds:

| | $\gamma=0.9995$ | $\gamma=1$ | difference | pairs slower at $\gamma=1$ |
|---|---:|---:|---:|---:|
| REINFORCE | 26.01 | 30.97 | $+4.96\pm2.47$ | 5/5 |
| A2C | 25.08 | 27.52 | $+2.45\pm0.72$ | 5/5 |
| PPO | 23.52 | 27.51 | $+3.99\pm2.60$ | 5/5 |

**Fifteen pairs out of fifteen**, in the same direction, across three algorithms
that share almost nothing in how they collect or update. A sign test gives
$p=3\times10^{-5}$. Given that seed variance has dominated every other
comparison in this project, this is about as unambiguous as the task produces.

## 4. What that costs, and what it buys

It is not a uniform loss. Undiscounted evaluation return — the quantity
$\gamma=1$ targets exactly — moves much less than the lap time does:

| | $\gamma=0.9995$ | $\gamma=1$ | difference | pairs |
|---|---:|---:|---:|---:|
| REINFORCE | 208.5 | 191.6 | $-16.8\pm8.9$ | 0/5 |
| A2C | 208.5 | 204.2 | $-4.3\pm7.6$ | 1/5 |
| PPO | 200.1 | 197.7 | $-2.5\pm23.6$ | 1/5 |

Only REINFORCE loses return outside its own confidence interval. A2C and PPO
break even, because what they lose in lap-time bonus they recover in
**reliability**:

| | $\gamma=0.9995$ | $\gamma=1$ | pairs favouring $\gamma=1$ |
|---|---:|---:|---:|
| A2C evaluation completion | 0.98 | 1.00 | 4/5 on progress |
| PPO evaluation completion | 0.90 | 0.96 | 3/5 on progress |
| PPO training completion rate | 0.7934 | 0.8237 | 5/5 |

So the honest summary is a **trade, not a regression**: without the discount the
agents drive slower and finish more often.

## 5. The mechanism is an implicit time incentive

The discount is not doing the textbook job. It is shaping the reward.

A terminal reward of roughly $R_{\text{finish}}+R_{\text{lap}}\approx145$ arrives
at the end of the lap and is weighted by $\gamma^{t}$. Finishing one hundred
steps sooner multiplies it by $\gamma^{-100}\approx1.05$, worth about $9.5$
points of discounted value. The *explicit* incentive to do the same thing — the
lap-time bonus $R_{\text{lap}}(1-t/T_{\max})$ — is worth $12.5$ points for the
same hundred steps.

**$\gamma=0.9995$ roughly doubles the time pressure the reward function was
designed to apply.** Remove it and the agents trade speed for caution, which is
exactly what all fifteen pairs show.

Two alternative explanations were tested and do not account for it:

* **Estimator variance.** Real for REINFORCE, whose Monte Carlo return has no
  trace: its coefficient of variation rises from $0.120$ to $0.155$, a $29\%$
  increase that survives normalizing by the mean, and its gradient
  signal-to-noise falls. This is why REINFORCE is the one algorithm that also
  loses return. But the channel is closed for A2C and PPO by advantage
  standardization, and they show the speed effect just as strongly.
* **Critic quality.** A2C's explained variance is $\approx0$ in *both* arms
  ($-0.013$ against $-0.008$): a critic that explains nothing either way cannot
  produce a 5/5 difference. PPO's does fall, from $0.160$ to $0.095$, which
  section 6 tests directly.

## 6. Control: is PPO's loss only a mis-tuned critic?

The grid varies $\gamma$ while holding the critic rate at $10^{-2}$, which was
calibrated at $\gamma=0.9995$. An undiscounted run regresses on targets $62\%$
wider, so that is no longer the same rate in any meaningful sense, and PPO's
falling explained variance is what a hot critic looks like. If a cooler critic
recovered the lost speed, this study would have measured a calibration mismatch
rather than the discount.

Three PPO seeds were therefore retrained at $\gamma=1$ with the critic rate
lowered to $6\times10^{-3}$ and to $3\times10^{-3}$, bracketing the $1/1.62$
rescale the wider targets imply.

| condition | n | speed | lap time | return | completion | explained variance |
|---|---:|---:|---:|---:|---:|---:|
| $\gamma=0.9995$, critic $10^{-2}$ | 5 | **21.35** | **23.52** | 200.1 | 0.90 | **0.160** |
| $\gamma=1$, critic $10^{-2}$ | 5 | 18.48 | 27.51 | 197.7 | 0.96 | 0.095 |
| $\gamma=1$, critic $6\times10^{-3}$ | 3 | 17.40 | 28.99 | 193.7 | 0.97 | 0.090 |
| $\gamma=1$, critic $3\times10^{-3}$ | 3 | 17.66 | 29.59 | 196.4 | **1.00** | 0.110 |

**A cooler critic does not recover the speed.** Every undiscounted variant sits
between $17.4$ and $18.5\,\mathrm{m/s}$, against $21.35$ with the discount, and
lowering the rate moves lap time the *wrong* way. Explained variance barely
responds either — $0.095\to0.110$ at best, nowhere near the discounted run's
$0.160$.

Two conclusions follow. The speed effect is the discount and not a calibration
artifact, which is what the control was for. And PPO's critic is genuinely
harder to fit against an undiscounted return at any of these rates, which is a
real if secondary cost of $\gamma=1$ rather than something a recalibration would
remove.

## 7. What to do for the reported experiments

**Keep $\gamma=0.9995$, and stop describing it as a discount.**

The evidence does not say $\gamma=1$ is broken — it converges, it completes more
laps, and for A2C and PPO it scores the same undiscounted return. It says the
two settings encode **different tasks**: with the discount, a racing task;
without it, a finishing task. Since the project is about racing, and since
`MDP.md` justifies its coefficients on the requirement that a fast lap beat a
slow one by a visible margin, the discounted setting is the one that matches the
stated intent.

Three consequences follow.

1. **Do not change $\gamma$ per algorithm or per actor size.** It is a task
   parameter here, not a tuning knob, and varying it would confound both
   experiment matrices.
2. **Record what it is.** `MDP.md` currently presents $\gamma$ as a discounted
   horizon chosen from the task timescale. That is not what it does. It supplies
   about half of the effective time pressure, and the reward coefficients in
   section "Choosing the coefficients" were validated *without* accounting for
   it — condition 3's margin of $0.313$ is computed on undiscounted returns and
   therefore understates the incentive the agent actually sees.
3. **$\gamma=1$ remains a legitimate separate comparison**, and is now a cheap
   one: the runs exist. If it is ever reported, it should be framed as a
   speed-versus-reliability trade rather than as a discount ablation.

## 8. Limitations

* One circuit, one actor size, one observation, five seeds. Experiment 2's
  multi-circuit setting is untested, and generalization pressure could change
  the reliability half of the trade.
* The learning rates were calibrated at $\gamma=0.9995$. Section 6 controls this
  for PPO's critic only; A2C's is uncontrolled, though its critic explains no
  variance in either arm.
* REINFORCE updates every eight completed episodes, so at a fixed interaction
  budget its number of optimizer updates is **endogenous**: slower laps mean
  longer episodes mean fewer updates ($\gamma=1$: 362–424; $\gamma=0.9995$:
  412–501). Cause and effect are entangled for that algorithm. Both arms
  plateau well before the budget ends, which argues against the gap being
  simply a starved run, but does not eliminate the confound.
* The reported effect is on settled policies. Nothing here says the discount
  matters for how *fast* the task is learned; first-completion times do not
  separate.

## Reproducing

```
python experiments/compare_discount_horizon.py run --shard 0 --shards 2
python experiments/compare_discount_horizon.py control --seeds 3
python experiments/compare_discount_horizon.py analyse
```

Runs land under `results/pre_experiment_configuration/discount_horizon/`;
the tables and `outputs/discount_horizon.png` come from the analyse step.
