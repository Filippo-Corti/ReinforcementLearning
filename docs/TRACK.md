# Environment

This file contains information on how the racing track is represented and how such representation should be used throughout the project.

## Track Representation

The circuit is represented internally as a closed, **arc-length-parametrized
centerline $\gamma$**. Given a distance $s$ along the centerline, $\gamma(s)$
returns a point on the 2D plane:

$$
\gamma(s) = (x(s), y(s)), \quad s \in [0,S_{\text{track}}), \qquad
\gamma(s+S_{\text{track}})=\gamma(s).
$$

At each distance $s$ along the circuit, we also care about:

* The associated heading $\psi(s)$, which is the direction the track is pointing at the location given by $s$. It can be measured as the tangent vector to $\gamma(s)$ in $(x(s), y(s))$:

    $$ \psi(s) = \text{atan2}\left(\frac{dy}{ds}, \frac{dx}{ds}\right) $$

* The **local curvature** $\kappa(s)=d\psi/ds$, which is a property of the
  track alone and is precomputed in the track table.

The Frenet observation uses a separate **curvature-preview summary**:

$$
\ell(v_t)=5+0.7v_t,
\qquad
\bar{\kappa}_t =
\frac{1}{\ell(v_t)}
\int_0^{\ell(v_t)}
\kappa((s_t+u)\bmod S_{\text{track}})\,du.
$$

At $v_t\in[0,70]m/s$, the lookahead varies from $5m$ to $54m$. It therefore
summarizes a longer section at high speed. This value cannot be stored as one
track-only table column because it depends on $v_t$, but it is cheap to compute
at runtime by integrating the preprocessed local-curvature table. Using the
integral, rather than wrapping one endpoint heading difference, avoids aliasing
when the preview contains more than $180°$ of cumulative turning. The stored local
curvature $\kappa(s)$ remains useful for validation, rendering and future
observation variants.

Version 0 uses a constant track width of $w=12m$. The width is stored in the
track file so it can later become track-specific or vary along $s$. The left and
right boundaries are:

$$ \text{boundary}_{\pm}(s) = \gamma(s) \pm \frac{w}{2}\hat{n}(s) $$

where $\hat{n}(s)$ is the unit vector pointing to the left of the tangent:

$$ \hat{n}(s) = \left(-\sin\psi(s), +\cos\psi(s)\right) $$

This single parametrization is the ground-truth object the environment holds; both
observation types (Frenet, LiDAR) and all derived quantities (*e.g.*, collision, progress) are
computed from it, rather than being independently implemented.

### Track Generation and Table Representation

Track geometry is generated using a seeded **checkpoint-and-smoothing
algorithm: random checkpoints are scattered around a circle with angular and
radial jitter. A periodic cubic spline passes through those checkpoints, and
the resulting closed curve is rescaled by less than half a requested sample
interval before being resampled at the configured constant arc-length spacing.
This construction requires no smoothing hyperparameters beyond the checkpoint
configuration below.

The proposed version 0 generation defaults are explicit configuration values:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_checkpoints` | 12 | Checkpoints distributed around the loop |
| `base_radius` | $50m$ | Radius before radial jitter |
| `radial_jitter` | $\pm25\%$ | Independent checkpoint-radius variation |
| `angular_jitter` | $\pm\frac{1}{4}$ sector | Variation from equally spaced checkpoint angles |
| $\Delta s_{\text{gen}}$ | $0.5m$ | Spacing of the final lookup table |
| $w$ | $12m$ | Constant version 0 track width |
| `max_attempts` | 100 | Deterministic retries before generation fails |

Every generation request requires an integer seed. Retries must derive their
random streams deterministically from that seed, so the same configuration and
seed produce byte-equivalent track data with the same generator version.

Generated tracks must satisfy all of the following:

1. The centerline is a simple closed curve: non-adjacent centerline segments do
   not intersect.
2. The seam is periodic in position, tangent and curvature. The final serialized
   table does not duplicate the first point; the closing segment is implicit.
3. Every sampled curvature respects the vehicle's kinematic minimum turning
   radius:

$$ \frac{1}{|\kappa(s)|} \ge R_{\min} = \frac{L}{\tan(\delta_{max})} \quad \forall s $$

4. The distance between non-neighbouring centerline segments is greater than
   $w+2m$, preventing overlapping boundaries and leaving a $1m$ margin on both
   sides.
5. The left and right boundaries are themselves simple closed curves and never
   intersect each other.
6. The total length is between $200m$ and $600m$. This configurable training
   scale is approximately one fifth of the original $1000m$--$3000m$ range.

The turning-radius check guarantees only kinematic steerability. It does not
guarantee that a future grip-limited car can take every corner at every speed.
If a generated candidate fails any validation, the generator retries until
`max_attempts` and then raises an error rather than returning an invalid track.

The final dense lookup table stores:

$$ \texttt{table}[i] = (s_i, x_i, y_i, \psi_i, \kappa_i) $$

In particular:

* $s_i=i\Delta s_{\text{gen}}$.
* $(x_i, y_i)$ are sampled from the final smoothed curve.
* $\psi_i$ is computed from the tangent of the final resampled curve.
* $\kappa_i$ is computed with periodic indexing:

    $$ \kappa_i = \frac{\text{wrap}(\psi_{i+1} - \psi_{i-1})}{2 \Delta s_{\text{gen}}} $$

  where `wrap` maps angular differences to $[-\pi,\pi)$.

### Persistent Track Format

Tracks are data rather than Python code and are stored under `tracks/*.json`.
Version 0 files contain:

```json
{
  "format_version": 1,
  "generation": {
    "seed": 0,
    "n_checkpoints": 12,
    "base_radius": 50.0,
    "radial_jitter": 0.25,
    "angular_jitter": 0.25,
    "max_attempts": 100
  },
  "width": 12.0,
  "sample_spacing": 0.5,
  "track_length": 300.0,
  "start_index": 0,
  "samples": [
    {"s": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "curvature": 0.0}
  ]
}
```

The sample shown is schematic; `track_length` and all samples come from the
generator. The loader decodes this project-owned format and rejects unknown
`format_version` values before applying the geometric constraints. The format
uses meters, radians and inverse meters throughout, as documented by the field
descriptions rather than by a redundant units object. Generation metadata is
retained even though runtime behaviour depends only on the validated geometry.

## Rendering

Version 0 presentation uses an 800×800 pixel Pygame canvas. Its camera fits both
track boundaries into the frame with 40 pixels of padding on every side and uses
a uniform scale, so geometric angles are visually preserved. The renderer draws
the road, boundaries, centerline, canonical finish gate and a triangular marker
whose nose indicates the car heading. These camera, canvas and colour choices
are display-only and do not affect the MDP state, dynamics, reward or episode
lifecycle.

## Track Usage

Runtime track preparation is intentionally separate from the Gymnasium
environment. `Track` owns the sampled, persistent circuit data, while
`TrackWithGeometry` combines it with periodic interpolation, sampled boundaries
and spatial indexes. Generate or load that prepared object before constructing
the environment:

```python
from envs import RacingEnv, TrackWithGeometry

generated_environment = RacingEnv(TrackWithGeometry.generate(seed=0))
saved_environment = RacingEnv(TrackWithGeometry.load("tracks/circuit.json"))
```

`RacingEnv` accepts only a `TrackWithGeometry`; it does not select between a raw
track, a file path and a generation seed. This keeps deterministic data
generation and persistence outside the simulation lifecycle and lets callers
reuse one prepared circuit across environments.

At each step, the car's Cartesian pose $(x_t, y_t)$ must be translated into the Frenet pose $(s_t, d_t)$, with:
* $s_t$ being the arc-length distance along the centerline between the start of the circuit and the closest centerline point to $(x_t, y_t)$.
* $d_t$ being the signed lateral offset, positive on the left side of the
  centerline and negative on the right.

We therefore need a way to compute the mapping:
$$ (x_t, y_t) \longrightarrow (s_t, d_t) $$

The projection is onto the closest point of a **centerline segment**, not merely
onto the closest sampled vertex. For a candidate segment from $p_i$ to
$p_{i+1}$, compute the clamped projection parameter

$$
u =
\operatorname{clip}_{[0,1]}
\frac{(p-p_i)^\top(p_{i+1}-p_i)}{\lVert p_{i+1}-p_i\rVert^2},
\qquad
p_{\text{proj}}=p_i+u(p_{i+1}-p_i).
$$

The associated $s$ is interpolated along that segment and wrapped to
$[0,S_{\text{track}})$. The signed lateral offset is

$$ d=(p-p_{\text{proj}})^\top\hat n(s). $$

To keep this cheap, use a temporally coherent segment window around the previous
projection. Its half-width in samples is
$\lceil v_{\max}\Delta t_{\mathrm{phys}}/\Delta s_{\mathrm{gen}}\rceil+4$.
Resets, teleports and failed local searches use a global spatial index over all
centerline segments. A local projection farther than
$w/2+v_{\max}\Delta t_{\mathrm{phys}}+4\Delta s_{\mathrm{gen}}$ from the query
point is treated as physically implausible and triggers the global fallback
rather than being accepted or silently clipped.

### Frenet Observation

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \delta_t, \bar{\kappa}_t) $$

computed as:

$$
\phi_{e,t} =
\operatorname{wrap}_{[-\pi,\pi)}(\theta_t-\psi(s_t)),
\qquad
\bar{\kappa}_t = \bar{\kappa}(s_t,v_t).
$$

Here $\theta_t$ is the car's heading from the true environment state, $\delta_t$
is the current front-wheel angle, and $\bar{\kappa}$ is the runtime preview
defined above. The environment exposes values in physical units; observation
normalization, if enabled for an agent, belongs in a wrapper or training utility
and must not alter the environment dynamics.

### LiDAR Observation

$$
o_t^{\text{LiDAR}} = (v_t, \delta_t, R_t), \qquad
R_t = (\tilde r_t^{(1)}, \dots, \tilde r_t^{(16)}).
$$

The ray offsets relative to car heading are inclusive and evenly spaced:

$$
\alpha_k=-100°+k\frac{200°}{15}, \qquad k=0,\ldots,15.
$$

Each ray starts at the point-car pose, has a maximum range
$r_{\max}=100m$, and returns the distance to its first boundary intersection.
A ray with no hit returns $r_{\max}$. The policy receives normalized ranges
$\tilde r=r/r_{\max}\in[0,1]$; raw metre values may be included in the `info`
dictionary for debugging.

Ray casting queries a spatial index over **all** left and right boundary
segments whose bounding boxes overlap the ray. It must not be restricted only
by arc-length proximity, because a ray can see a geometrically nearby section
that is far away along the lap. LiDAR reflects only solid track boundaries:
there is no separate off-road buffer layer.

### Collision Detection

Given the projection $(s_t, d_t)$, the car is off-track if:

$$ |d_t| \ge \frac{w}{2}. $$

This rule treats the car as a point and is evaluated after every physics
substep. A finite vehicle footprint is explicitly deferred as described in
[`MDP.md`](MDP.md).

Most crashes now arrive through this rule rather than through a dedicated
physical failure. With the tyre friction budget in force, entering a corner too
fast does not spin the car: it understeers, the achieved yaw rate falls short of
the requested one, and the resulting path runs wide until $|d_t|$ reaches the
boundary. The physics decides where the car goes; the geometry decides that
going there ends the episode.

### Progress Term of the Reward

Let $s_t^{\text{wrap}}\in[0,S_{\text{track}})$ be the projected position. The
signed change across the periodic seam is

$$
\Delta s_t =
\operatorname{wrap}_{[-S_{\text{track}}/2,S_{\text{track}}/2)}
\left(s_t^{\text{wrap}}-s_{t-1}^{\text{wrap}}\right).
$$

This is positive when moving forward and negative when moving backward. It is
unwrapped, but **not forced to be monotonic**:

$$
s_t^{\text{episode}}=s_{t-1}^{\text{episode}}+\Delta s_t,
\qquad
s_0^{\text{episode}}=0.
$$

Because the vehicle cannot travel half a circuit in one step, the periodic wrap
has an unambiguous branch. An implausibly large $|\Delta s_t|$ indicates a
projection failure and triggers global reprojection. The normalized reward term
is

$$ \Delta\tilde{s}_t = \frac{\Delta s_t}{S_{\text{track}}}, $$

with $\Delta\tilde{s}_0=0$ on reset.

### Finish Line and Lap Completion

The finish gate of an episode is the segment joining the two boundaries at the
arc length the car started from, oriented by the forward tangent there. A
forward crossing requires the car trajectory to intersect this segment and have
positive motion along that tangent.

Because the start pose is sampled (see [`MDP.md`](MDP.md)), the gate moves with
it: the objective is always one full circuit from wherever the car was placed,
never a partial run to a line somewhere else on the track. With a fixed start the
gate coincides with `start_index`, recovering the canonical finish line. The
geometry is resolved once at reset rather than at every physics substep.

An episode finishes only when all of the following hold:

1. The car crosses the finish gate in the forward direction.
2. It is still on track after the crossing.
3. Its signed episode progress is at least
   $S_{\text{track}}-\epsilon_{\text{finish}}$, where
   $\epsilon_{\text{finish}}=\max(2\Delta s_{\text{gen}},
   v_{\max}\Delta_{t_{agent}})=2.8m$ with the current constants.

The progress requirement prevents the reset pose on the finish line, local
oscillation across the gate, or a nearby geometric shortcut from completing a
lap. If collision and finish are detected in the same physics substep,
collision takes precedence.
