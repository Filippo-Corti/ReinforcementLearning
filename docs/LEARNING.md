# Learning Contract

This file fixes the mathematical and numerical contract for the three
project-owned learning algorithms. The notation follows the course notes in
`docs/theory/`; the equations below specialize it to the racing observations,
bounded two-dimensional actions and Gymnasium episode boundaries.

## Notation and objective

At agent timestep $t$, the policy receives an observation $O_t$ and chooses

$$
A_t=(A_t^{\mathrm{throttle}},A_t^{\mathrm{steer}})\in[-1,1]^2.
$$

The environment returns $R_{t+1}$, $O_{t+1}$, `terminated` $z_t$ and
`truncated` $q_t$. A finish or crash has $z_t=1$; the $5000$-step time limit has
$q_t=1$. The learning objective is the expected discounted return

$$
J(\mathbf{\theta})=
\mathbb E_{\pi_{\mathbf{\theta}}}
\left[\sum_{t=0}^{\tau}\gamma^t R_{t+1}\right],
\qquad \gamma=0.9995,
$$

where $\tau$ is the final transition of the episode. Raw undiscounted episode
return remains the reported task metric.

The actor has parameters $\mathbf{\theta}$. A2C and PPO additionally use the
critic $v_{\mathbf w}(O_t)$ with parameters $\mathbf w$. Quantities used as
targets or policy-gradient weights are detached from the graph.

## Bounded Gaussian policy

The actor MLP returns the mean $\boldsymbol\mu_{\mathbf\theta}(O_t)\in
\mathbb R^2$ of a diagonal Gaussian. Its state-independent learned
$\boldsymbol\ell\in\mathbb R^2$ is initialized to $-0.5$ and constrained to
$[-5,2]$ when computing
$\boldsymbol\sigma=\exp(\boldsymbol\ell)$. For independent
$\varepsilon_t\sim\mathcal N(\mathbf 0,I)$,

$$
U_t=\boldsymbol\mu_{\mathbf\theta}(O_t)
  +\boldsymbol\sigma\odot\varepsilon_t,
\qquad
A_t=\tanh(U_t).
$$

Thus the policy is fully mixed on the interior $(-1,1)^2$. Deterministic
evaluation does not sample and uses

$$
A_t^{\mathrm{eval}}=\tanh
\left(\boldsymbol\mu_{\mathbf\theta}(O_t)\right).
$$

Policy sampling uses the dedicated policy RNG. The sampled $U_t$ and $A_t$ are
treated as constants when evaluating score-function losses; gradients do not
flow through the sampling path.

The log-probability of a vector action is one scalar, summed over its two
components:

$$
\log\pi_{\mathbf\theta}(A_t\mid O_t)=
\sum_{j=1}^{2}
\left[
\log\mathcal N(U_{t,j};\mu_{\mathbf\theta,j}(O_t),\sigma_j^2)
-\log(1-\tanh^2(U_{t,j}))
\right].
$$

The Jacobian term is evaluated stably as

$$
\log(1-\tanh^2 u)=
2\left(\log 2-u-\operatorname{softplus}(-2u)\right).
$$

Sampling retains $U_t$. When only $A_t$ is available, each component is first
clamped to $[-1+10^{-6},1-10^{-6}]$ and inverted with
$U_t=\operatorname{atanh}(A_t)$. No action is clipped after sampling. The
sample estimate $-\log\pi_{\mathbf\theta}(A_t\mid O_t)$ is logged as a
transformed-policy entropy proxy; its coefficient in every actor loss is zero.

## Shared model and numerical choices

| Concern | Contract |
|---|---|
| Actor hidden sizes | Experiment factor: `(32, 32)`, `(64, 64)`, `(256, 256)` |
| Critic hidden sizes | Fixed `(64, 64)` |
| Hidden activation | `Tanh` |
| Initialization | Orthogonal weights, zero biases; gain $\sqrt 2$ in hidden layers, $0.01$ for the actor mean output and $1$ for the critic output |
| Numerical type | PyTorch `float32` |
| Observation normalization | Per-component running mean and population variance; divide by $\sqrt{\mathrm{variance}+10^{-8}}$, then clamp to $[-10,10]$ |
| Optimizers | Separate Adam optimizers; $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, no weight decay or scheduler |
| Gradient clipping | Actor and critic norms clipped separately to $0.5$ |
| Loss reduction | Explicit means below; no implicit extra value or entropy coefficient |

Running observation statistics are updated only from training observations.
Each current training observation updates the statistics before it is
normalized. A next observation needed only for bootstrap uses the resulting
statistics without updating them; it updates the statistics later only if it
becomes the current observation. The normalized current and bootstrap inputs
are retained with the rollout, so later PPO epochs see the same values.
Evaluation and reference runs use frozen statistics.

The fixed collection and update settings are:

| Algorithm | Collection | Update settings |
|---|---|---|
| REINFORCE | 8 complete episodes | one actor update per batch |
| A2C+GAE | 2048 transitions | one full-rollout actor update and one critic update; $\lambda=0.95$ |
| PPO | 2048 transitions | minibatches of 256, 10 epochs, clip $\epsilon=0.2$, no KL early stop and no value clipping |

Only learning rates are calibrated. The finite candidates are:

- REINFORCE actor: $\{10^{-4},3\cdot10^{-4},10^{-3}\}$;
- A2C actor/critic: $\{(10^{-4},3\cdot10^{-4}),
  (3\cdot10^{-4},10^{-3})\}$;
- PPO actor/critic: $\{(10^{-4},3\cdot10^{-4}),
  (3\cdot10^{-4},10^{-3})\}$.

The selection procedure is fixed in `EXPERIMENT.md`. No network size receives
its own learning-rate calibration.

## Episode boundaries and critic targets

Two masks give the boundary semantics explicitly:

$$
b_t=1-z_t,
\qquad
c_t=(1-z_t)(1-q_t)(1-h_t),
$$

where $b_t$ permits value bootstrap and $c_t$ permits recursive credit to the
next stored transition. The rollout-cut indicator $h_t$ is one only on the last
record of a fixed rollout; it is zero for ordinary transitions.

The TD error is

$$
\delta_t^{\mathbf w}=
R_{t+1}+\gamma b_t v_{\mathbf w}(O_{t+1})
-v_{\mathbf w}(O_t).
$$

Consequently, a true termination has no bootstrap. A time-limit truncation and
an artificial rollout cut bootstrap from $O_{t+1}$, but neither lets the GAE
recursion enter an unrelated reset episode or pass beyond the collected batch.

Working backward through a rollout,

$$
\hat{\mathbb A}_t=
\delta_t^{\mathbf w}
+\gamma\lambda c_t\hat{\mathbb A}_{t+1},
\qquad
y_t=\operatorname{stop}
\left(\hat{\mathbb A}_t+v_{\mathbf w}(O_t)\right).
$$

The raw $\hat{\mathbb A}_t$ defines $y_t$. For the actor only, advantages are
standardized once over the rollout using population standard deviation and
$10^{-8}$; PPO reuses those fixed standardized values for every epoch.

## REINFORCE

For complete trajectory $i$, the discounted return-to-go is computed backward:

$$
G_t^i=R_{t+1}^i+
\gamma(1-z_t^i)(1-q_t^i)G_{t+1}^i.
$$

REINFORCE has no critic, so both completion/crash and the environment time limit
end its Monte Carlo return. Across all transitions in the batch, define
$\widetilde G_t^i=(G_t^i-\overline G)/(s_G+10^{-8})$. This is the documented
batch baseline and scale; it does not introduce a learned value function. The
actor minimizes the trajectory-mean loss

$$
\mathcal L_{\mathrm{REINFORCE}}(\mathbf\theta)=
-\frac{1}{n}\sum_{i=1}^{n}\sum_{t=0}^{\tau_i}
\log\pi_{\mathbf\theta}(A_t^i\mid O_t^i)
\operatorname{stop}(\widetilde G_t^i).
$$

```text
initialize actor parameters theta
repeat:
    collect n = 8 complete trajectories with pi_theta
    compute every discounted return-to-go G_t backwards
    standardize the batch of G_t values
    form L_REINFORCE from summed trajectory score functions
    update theta once with Adam and actor gradient clipping
```

## A2C with GAE

A2C uses the TD errors and GAE targets defined above. With $N=2048$ rollout
transitions, it minimizes separate mean losses

$$
\mathcal L_{\mathrm{actor}}(\mathbf\theta)=
-\frac{1}{N}\sum_{t=0}^{N-1}
\log\pi_{\mathbf\theta}(A_t\mid O_t)
\operatorname{stop}(\widetilde{\mathbb A}_t),
$$

$$
\mathcal L_{\mathrm{critic}}(\mathbf w)=
\frac{1}{2N}\sum_{t=0}^{N-1}
\left(v_{\mathbf w}(O_t)-y_t\right)^2.
$$

```text
initialize actor theta and critic w
repeat:
    collect 2048 transitions with pi_theta
    compute delta_t, GAE advantages and critic targets backwards
    standardize advantages for the actor only
    update theta once from L_actor with w and advantages detached
    update w once from L_critic with targets detached
```

## PPO

PPO stores the behaviour-policy log-probability
$\log\pi_{\mathbf\theta_{\mathrm{old}}}(A_t\mid O_t)$ at collection time. During
optimization,

$$
\omega_{\mathbf\theta/\mathbf\theta_{\mathrm{old}},t}
=\exp\left(
\log\pi_{\mathbf\theta}(A_t\mid O_t)
-\log\pi_{\mathbf\theta_{\mathrm{old}}}(A_t\mid O_t)
\right).
$$

For minibatch $B$, the clipped actor loss is

$$
\mathcal L_{\mathrm{actor}}(\mathbf\theta)=
-\frac{1}{|B|}\sum_{t\in B}
\min\left\{
\omega_t\widetilde{\mathbb A}_t,
\operatorname{clip}(\omega_t,1-\epsilon,1+\epsilon)
\widetilde{\mathbb A}_t
\right\},
$$

and the critic loss is

$$
\mathcal L_{\mathrm{critic}}(\mathbf w)=
\frac{1}{2|B|}\sum_{t\in B}
\left(v_{\mathbf w}(O_t)-y_t\right)^2.
$$

The old log-probabilities, advantages and targets remain fixed for all ten
epochs. Each epoch uses an independently seeded permutation and covers every
rollout row exactly once. Approximate KL is logged as
$|B|^{-1}\sum_{t\in B}(\log\pi_{\mathrm{old},t}-\log\pi_{\mathbf\theta,t})$.

```text
initialize actor theta and critic w
repeat:
    set theta_old <- theta and collect 2048 transitions
    retain old log-probabilities and compute fixed GAE targets
    for 10 epochs:
        generate a seeded permutation and split it into minibatches of 256
        for each minibatch:
            compute omega_theta/theta_old and the clipped actor loss
            update theta with old quantities detached
            update w from the unclipped critic loss
    discard the rollout before collecting with the updated policy
```
