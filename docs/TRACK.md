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

A circuit is built the way a real one is described: **a sequence of straights
and constant-radius corners**, rather than one smooth closed curve.

An earlier generator passed a periodic cubic spline through jittered polar
checkpoints. A spline through points has continuously varying curvature
everywhere by construction, so it can never hold $\kappa=0$ for the length of a
straight, and measurement bore this out: only $3.7\%$ of a generated lap was
straighter than $500\,\mathrm m$ of radius while $81.6\%$ of it was curving at
$100\,\mathrm m$ or tighter. The result was a wobbly ring with nothing to brake
for, because there was nowhere to build speed. The generator below produces
circuits that are about $63\%$ straight.

#### The construction

**1. Sample a closed polygon.** `n_corners` vertices are placed around one
centre at evenly spaced angles with jitter, and at `base_radius` with radial
jitter. Sorting the vertices by angle makes the polygon *star-shaped* and so
guaranteed simple, which is what lets the next step assume that only
neighbouring edges can interfere. This polygon is the route the circuit takes;
it is never part of the final track.

**2. Round off every vertex into a corner.** At a vertex whose direction changes
by $\Delta$, a corner of radius $R$ meets each of the two edges at a **tangent
distance**

$$ T = R\tan\left(\frac{|\Delta|}{2}\right) $$

from the vertex, and sweeps an arc of length $R|\Delta|$. The corner radius and
the surrounding straights therefore trade off directly: a larger radius eats
further back along both edges. Each corner may claim at most **half** of each
edge it touches, which leaves every straight non-negative without any search.

**3. Choose each radius as a fraction of what fits.** The radius is drawn as
$R=\rho R_{\max}$, where $R_{\max}$ is the largest radius the two adjacent edges
admit and $\rho\sim\mathcal U(\texttt{corner\_radius\_fraction})$, then clipped
to $[\texttt{min\_corner\_radius},\texttt{max\_corner\_radius}]$. Drawing a
*fraction* rather than an absolute length is what makes the contrast
controllable: a small $\rho$ gives a tight corner between long straights and a
large $\rho$ gives a sweeper, on a vertex of any size. A vertex too sharp for
the edges meeting it, so that $R_{\max}$ falls below the minimum radius, rejects
the candidate.

**4. Emit the segment list.** Walking the polygon gives, for each vertex in
turn, its arc followed by whatever survives of the outgoing edge:

$$ \text{straight}_i = \lVert e_i\rVert - T_i - T_{i+1} \ \ge 0. $$

Because both corners keep their tangent points *on* the polygon edges, the
circuit closes exactly when the polygon does. No numerical closure solve is
needed, which is the reason for building on a polygon rather than integrating a
sampled curvature profile.

**5. Scale to a whole number of samples.** The total length is rescaled by
under half a sample interval so it is an exact multiple of
$\Delta s_{\text{gen}}$. Scaling every radius and every straight by the same
factor is a similarity transform, so closure survives it. The radius band of
step 3 is applied before this rescale, so a finished corner may sit a fraction
of a percent outside it.

**6. Put the seam in the middle of the longest straight.** The start and finish
line then sits where a real one does, and the seam joins two samples that both
have zero curvature, so the periodic table is continuous in position, heading
and curvature with no special handling.

#### Defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `n_corners` | 9 | Corners, and so polygon vertices |
| `base_radius` | $70m$ | Radius the polygon is sampled around |
| `radial_jitter` | $\pm55\%$ | Independent vertex-radius variation |
| `angular_jitter` | $\pm\frac{3}{10}$ sector | Variation from equally spaced vertex angles |
| `corner_radius_fraction` | $[0.25,0.80]$ | Corner radius as a fraction of the largest that fits |
| `min_corner_radius` | $12m$ | Tightest corner the generator may produce |
| `max_corner_radius` | $200m$ | Most open corner the generator may produce |
| $\Delta s_{\text{gen}}$ | $0.5m$ | Spacing of the final lookup table |
| $w$ | $12m$ | Constant version 0 track width |
| `max_attempts` | 100 | Deterministic retries before generation fails |

The radial jitter is large on purpose. At $\pm35\%$ a fifth of generated
circuits had *no right-hand corner at all*, because a polygon that stays close
to its circle stays convex and every vertex then turns the same way. At
$\pm55\%$ some vertices turn inward, every circuit contains corners in both
directions, and $26\%$ of all corners are right-handers.

With these defaults, 60 of 60 consecutive seeds generate a valid circuit of
$414$ to $756\,\mathrm m$, about $63\%$ of it straight, with corner radii whose
median is $31\,\mathrm m$ and whose tenth and ninetieth percentiles are $12$ and
$104\,\mathrm m$.

Every generation request requires an integer seed. Retries derive their random
streams deterministically from that seed, so the same configuration and seed
produce byte-equivalent track data with the same generator version.

#### What this model omits

Real circuits ease into a corner along a **clothoid**, a transition whose
curvature grows linearly with distance, instead of stepping from zero curvature
to $1/R$ at a point. This generator makes that step. The consequence is bounded
and visible: curvature is piecewise constant, so $\kappa$ jumps at a corner's
entry and exit. It does not make the circuit undrivable, because the steering
rate limit of $180°\,\mathrm s^{-1}$ already prevents the car from following any
such step instantly, and the preview curvature $\bar\kappa_t$ in the Frenet
observation is an integral and therefore stays continuous. Transitions are a
refinement this project has not needed.

The polygon is star-shaped, so a circuit cannot fold back on itself the way
Monaco does. Every ray from the centre crosses the track exactly once.

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
6. The total length is between $300m$ and $700m$. This configurable training
   scale keeps a lap well inside the $T_{\max}$ episode cap.

The turning-radius check guarantees only kinematic steerability. It does not
guarantee that the grip-limited car can take every corner at every speed. If a
generated candidate fails any validation, the generator retries until
`max_attempts` and then raises an error rather than returning an invalid track.

Rule 4 needs one qualification, because it would otherwise reject tight corners
rather than the folding it is meant to catch. The straight-line distance across
an arc is shorter than the arc itself, and at the minimum corner radius it drops
below $w+2m$ while the corner is still turning. The rule therefore ignores pairs
of samples closer together along the track than $\pi\,\texttt{min\_corner\_radius}$,
half a turn at the tightest permitted radius, within which a corner cannot have
come back around towards itself.

The final dense lookup table stores:

$$ \texttt{table}[i] = (s_i, x_i, y_i, \psi_i, \kappa_i) $$

Every column is evaluated in closed form from the segment the sample falls in,
rather than integrated numerically and then differenced:

* $s_i=i\Delta s_{\text{gen}}$.
* $(x_i, y_i)$ and $\psi_i$ come from advancing the segment's start pose by the
  sample's offset into it. A straight advances along its heading; an arc of
  signed curvature $\kappa$ turns about a centre offset $1/\kappa$ to its left,
  so one expression covers left-hand and right-hand corners.
* $\kappa_i$ is the curvature of that segment: exactly $0$ on a straight and
  exactly $\pm1/R$ in a corner.

This is the practical gain of the segment construction over a spline. The
previous generator inverted arc length numerically to place its samples and then
recovered curvature by differencing neighbouring headings; both steps are now
unnecessary, and $\kappa$ is exact rather than a finite difference.

### Persistent Track Format

Tracks are data rather than Python code and are stored under `tracks/*.json`.
Version 0 files contain:

```json
{
  "format_version": 1,
  "generation": {
    "seed": 0,
    "n_corners": 9,
    "base_radius": 70.0,
    "radial_jitter": 0.55,
    "angular_jitter": 0.3,
    "max_attempts": 100
  },
  "width": 12.0,
  "sample_spacing": 0.5,
  "track_length": 500.0,
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

Presentation uses an 800×800 pixel Pygame canvas in one of two styles, chosen by
`RacingEnv(render_style=...)`. Both are handed the same immutable frame — pose,
finish-gate arc length, elapsed time, progress and the last applied action — and
neither can reach back into the simulation. Camera, canvas and colour choices
are display-only and do not affect the MDP state, dynamics, reward or episode
lifecycle.

**Minimal** answers *where did the car go*. It fits both boundaries into the
frame at a uniform scale, so geometric angles are visually preserved, and draws
the road, its edges, the finish line and a dot. Flat colours and no text, so two
frames can be compared at a glance.

**Broadcast** answers *what was it like to drive*. The main image is a
perspective projection of the road ahead from the car's own pose: the boundaries
are sampled forward along the centerline, expressed in car-relative metres and
projected through a pinhole at a fixed eye height, giving a level horizon that
holds still while the road turns beneath it. Distant road is blended toward the
horizon, which reads as depth and hides the end of the drawn lookahead. Kerbs,
centre dashes and the finish line are keyed to arc length rather than to the
sampling, so they stay attached to the track and stream past the car. A corner
inset carries the circuit from above, because a forward view cannot show the
shape of the lap or where in it the car is. The overlay reports speed in km/h,
lap time, lap progress, throttle, brake and steering.

### The finish line is the episode's, not the circuit's

Both styles draw the finish line at **this episode's** gate arc length, taken
from the episode lifecycle. A lap runs one full circuit from wherever the car is
placed, so an episode that begins at a sampled start also ends there. Drawing
the canonical start line instead would mark a place that episode never treats as
a finish — which is what the earlier renderer did.

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

The offset is measured the same way as every other angle in this project, so a
positive $\alpha$ turns toward the left of travel: ray $0$ looks $100°$ to the
right and ray $15$ looks $100°$ to the left. Nothing behind the car within
$80°$ of straight back is seen at all.

Each ray starts at the point-car pose, has a maximum range
$r_{\max}=100m$, and returns the distance to its first boundary intersection.
A ray with no hit returns $r_{\max}$. The policy receives normalized ranges
$\tilde r=r/r_{\max}\in[0,1]$; raw metre values may be included in the `info`
dictionary for debugging.

Ray casting tests **all** left and right boundary segments, in one vectorized
pass over every (ray, segment) pair. It must not be restricted by arc-length
proximity, because a ray can see a geometrically nearby section that is far
away along the lap.

The only segments discarded are those whose distance from the vehicle exceeds
$r_{\max}$ plus half a segment, which no ray could reach whatever its
direction. That test is a distance in space, not a position along the lap, so
it never hides the case above. A sampled circuit has a few thousand segments,
which measures at about $0.14\,\mathrm{ms}$ per observation — roughly five
minutes of processor time across a two-million-step run, spread over the
collection workers. A spatial index would cost more bookkeeping than it saves
at this size.

LiDAR reflects only solid track boundaries: there is no separate off-road
buffer layer.

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
