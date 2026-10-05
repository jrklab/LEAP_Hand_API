# LEAP Hand Python Scripts — Guide

Covers everything in this `python/` folder beyond the base SDK: motor calibration, the
webcam gesture-mimic demo, and the experimental walking gait. For hardware setup (USB
connection, serial permissions) see `../readme.md` first.

All commands below assume:

```bash
cd LEAP_Hand_API/python
source test_env/bin/activate   # or whatever venv has dynamixel_sdk/numpy/opencv/mediapipe
```

---

## 1. Architecture

### 1.1 Control stack

```
dynamixel_sdk (vendor SDK, raw register read/write)
        |
leap_hand_utils/dynamixel_client.py : DynamixelClient
        |   - works in RADIANS, converts to/from raw ticks internally
        |   - sync_write / set_torque_enabled / write_desired_pos
        |   - read_pos / read_vel / read_cur / read_pos_vel / read_pos_vel_cur
        v
hand_demo.py : LeapNode
        |   - owns a DynamixelClient, sets PID gains + current limit on connect
        |   - set_leap(pose)     : pose already in raw LEAPhand radians
        |   - set_allegro(pose)  : pose in allegro-zero-centered radians, converts for you
        v
higher-level scripts: hand_demo.py's own main(), leap_hand_walk.py
```

`DynamixelClient` was rewritten once (see git history commit "Restore DynamixelClient
driver...") after an earlier commit had replaced it with a lower-level `MotorDriver`
class that `hand_demo.py`/`main.py` didn't actually use — if you ever see
`NameError: DynamixelClient`, that regression is back; the class lives in
`leap_hand_utils/dynamixel_client.py`.

### 1.2 Angle conventions

Three different "zero" conventions show up across these scripts — all in
`leap_hand_utils/leap_hand_utils.py`:

| Convention | 0 means | Used by |
|---|---|---|
| **raw ticks** | motor's physical zero (4096 ticks/rev) | `dynamixel_client.py` internals only |
| **LEAPhand radians** | *(none — π is the reference, not 0)* | `write_desired_pos`, `read_pos`, `LeapNode.set_leap`, `angle_safety_clip` |
| **allegro radians/degrees** | fully open/straight pose, for every joint including the thumb | `LeapNode.set_allegro`, all the per-joint tuning constants in `hand_joint_tracker.py` and `leap_hand_walk.py` |

The fixed relationship: **raw tick 2048 = π rad (LEAPhand) = 0 rad (allegro) = the
documented "fully open" pose**, for every joint. `lhu.allegro_to_LEAPhand(x)` just adds
π; `lhu.LEAPhand_to_allegro(x)` subtracts it. `lhu.angle_safety_clip(x)` clips **LEAPhand**
radians against `lhu.LEAPsim_limits()` (which are themselves expressed in allegro-space,
shifted by `LEAPsim_to_LEAPhand`). When writing new code: build poses in allegro
degrees/radians (they're intuitive — 0 = neutral, positive = curl further closed), then
run `allegro_to_LEAPhand(..., zeros=False)` → `angle_safety_clip(...)` before calling
`write_desired_pos`/`set_leap`. `leap_hand_walk.py`'s `_to_real_radians()` is exactly this
pipeline and is the pattern to copy for any new script.

### 1.3 Motor ID / joint layout

IDs 0–15, in blocks of 4: **Index (0–3), Middle (4–7), Ring (8–11), Thumb (12–15)**, each
block ordered **MCP_Side, MCP_Forward, PIP, DIP**. E.g. Index MCP_Side = ID 0, Ring
MCP_Forward = ID 9, Thumb DIP = ID 15. MCP_Side is the ab/adduction ("spread") joint;
real per-joint ranges (allegro-degrees, from `lhu.LEAPsim_limits()`, matching
`leap-hand/LEAP_Hand_Sim`'s URDF) are roughly:

| Joint | Index/Middle/Ring range | Thumb range |
|---|---|---|
| MCP_Side (ab/adduction) | −18° to +60° (±60° about 0) | −20° to +120° |
| MCP_Forward (main flex) | −18° to +128° | −27° to +140° |
| PIP | −29° to +108° | −69° to +109° |
| DIP | −21° to +117° | −77° to +108° |

The thumb's joint semantics don't cleanly match "Side/Forward/PIP/DIP" the way the other
three fingers do — its first joint is closer to a CMC opposition joint. Treat its
per-joint behavior as something to discover empirically (see §4.4), not assume from the
naming.

---

## 2. `calibrate_motor_offsets.py` — mechanical mount calibration

**Problem it solves:** a servo horn mounted 90°/180°/270° off from correct (a common
assembly mistake — the spline doesn't visually indicate which tooth is "zero").

**How:** writes directly to each motor's `Homing_Offset` register (control table address
20, signed, EEPROM-area). The motor firmware applies this correction to every future
`Present_Position` read and `Goal_Position` write, so no client code needs special-casing
for a miscalibrated motor.

**Algorithm:** for each motor, read the *true* raw sensor value (current `Present_Position`
minus whatever `Homing_Offset` is already set, so it's safe to re-run), compute its offset
from the expected straight-pose reading of 2048 ticks, **snap that offset to the nearest
90° (1024-tick) multiple**, and write the negated snapped value as the new
`Homing_Offset`. Snapping (rather than using the raw measured offset) matters because a
real mount error only ever happens in 90° spline increments — any smaller residual is just
imprecision in how exactly "straight" you held the finger, not a real mechanical offset,
and shouldn't be compensated away.

**Usage** — hold the motors you're calibrating at the fully straight/extended reference
pose, then:

```bash
python calibrate_motor_offsets.py --motors 2 10 11            # calibrate these IDs
python calibrate_motor_offsets.py --motors 2 10 11 --dry-run   # preview only
python calibrate_motor_offsets.py --motors 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  # all
```

**Tuning:** `EXPECTED_STRAIGHT_TICKS = 2048` and `STEP = 1024` (90°) are fixed by the
hardware/convention, not meant to be tuned. `--port` if your hand isn't on
`/dev/ttyUSB0`.

---

## 3. `hand_demo.py` — direct control + gesture mimic

### 3.1 `LeapNode` class

Thin wrapper around `DynamixelClient`: on construction it sets position-current control
mode, PID gains (`kP=600, kI=0, kD=200`, with MCP_Side joints at 0.75× that for a softer
side-to-side feel), current limit (`curr_lim=350` mA — raise to 550 if you have the Full
hand, not the Lite), then ramps from wherever the motors currently are to the "all open"
pose over 32 steps (~1s) before returning. `set_leap(pose)` takes raw LEAPhand radians;
`set_allegro(pose)` takes allegro radians/convention and converts for you — prefer
`set_allegro` unless you have a specific reason not to.

### 3.2 Running it

```bash
python hand_demo.py --mode test        # triangle wave on all 16 motors, no camera needed
python hand_demo.py --mode realtime    # live control from hand_joint_tracker.py's UDP feed
```

`--mode test` is the quick "did my wiring/calibration survive" smoke test — always try
this before `realtime` after any hardware change. `--mode realtime` listens on UDP
`0.0.0.0:5005` for packets from `hand_joint_tracker.py` (§4) and must be run alongside it.

### 3.3 Realtime control pipeline

```
hand_joint_tracker.py --[UDP, 15 comma-separated ints]--> hand_demo.py
                                                              |
                                                   background thread (udp_listener)
                                                   parses packet -> target_pos
                                                              |
                                                   main loop, fixed 100Hz
                                                   (CONTROL_PERIOD_S = 0.01)
                                                              |
                                      pos = slew_limit(pos, target_pos, max_step)
                                                              |
                                                   leap_hand.set_leap(pos)
```

The UDP receive and the motor-command loop are **deliberately decoupled** (a background
thread owns the socket; the main loop just reads whatever `target_pos` it last saw) so a
slow/irregular camera frame doesn't stall motor commands — the control loop keeps running
at a steady rate and slews toward the last known target instead. `slew_limit()` (shared
with `leap_hand_walk.py`) caps how far a commanded position can move per control tick, as
a backstop against a single garbled/noisy packet producing a sudden jump.

UDP packet format (15 ints): 12 flexion values (`Thumb MCP/PIP/DIP, Index MCP/PIP/DIP,
Middle ..., Ring ...` — degrees, already through `map_to_motor`), then 3 abduction values
(`Index, Middle, Ring` — signed degrees). See §4.2 for how these are produced.

### 3.4 Tuning knobs

| Constant | Location | Effect |
|---|---|---|
| `kP`, `kI`, `kD` | `LeapNode.__init__` | Stiffness/damping. Raise `kP` if weak, lower if jittery. |
| `curr_lim` | `LeapNode.__init__` | Max motor current (mA) — 350 (Lite) / 550 (Full). |
| `ABDUCTION_GAIN` | `main()`, realtime branch | Flip to `-1.0` if finger spread comes out mirrored (e.g. a V-gesture closes instead of spreads). |
| index/thumb per-finger multipliers (`*1.5`, `*1.8`, etc.) | `udp_listener()` | Ad hoc per-finger gain tweaks on top of the tracked angle — hand-tuned, not derived. |
| `CONTROL_PERIOD_S` | realtime branch | Control loop rate (currently 100Hz). |
| `MAX_SLEW_RAD_PER_S` | realtime branch | Safety backstop on commanded-position rate of change. |
| `PRINT_EVERY_N_LOOPS` | both branches | Console status print throttle. |

---

## 4. `hand_joint_tracker.py` — webcam → joint angles

Reads a webcam via MediaPipe Hands, estimates per-joint flexion + finger-spread angles,
maps them to motor-degree ranges, and sends them over UDP to `hand_demo.py --mode
realtime`. Needs a venv with `mediapipe==0.10.21` on **Python 3.12** specifically — the
legacy `mp.solutions.hands` API this script uses doesn't exist in `mediapipe>=1.0` and was
never built for Python 3.13 wheels at all.

```bash
python hand_joint_tracker.py
```

### 4.1 Per-joint flexion angle estimation

For each of Thumb/Index/Middle/Ring, three vertex angles are computed directly from
MediaPipe's 21 hand landmarks (`get_angle(a, b, c)` = angle at `b` between `a` and `c`):
MCP angle (`wrist, MCP, PIP`), PIP angle (`MCP, PIP, DIP`), DIP angle (`PIP, DIP, TIP`).
For the thumb these use its own landmark chain (CMC, MCP, IP, TIP) — its "MCP angle" uses
the wrist as a stand-in vertex since the thumb doesn't have a true adjacent joint there,
which is part of why its signal is noisier (see §4.3).

Each raw angle is smoothed with a `OneEuroFilter` (adaptive: more smoothing when the
signal is nearly still, less when moving fast — see the class docstring), then passed
through `map_to_motor(angle, open_deg, closed_deg, max_motor_deg=90)`, which linearly maps
the real/physical open→closed angle range onto a 0–90° motor-degree range.

### 4.2 Abduction (finger spread) estimation

Computed only for Index/Middle/Ring (the thumb's spread is mechanically distinct —
opposition across the palm, not spread between fingers — and isn't modeled). Builds a
palm-plane normal from `wrist`/`index_MCP`/`ring_MCP`, projects each finger's
`MCP→PIP` vector onto that plane, and measures its signed angle (`signed_abduction_angle`)
against the middle finger's projected vector as the zero-spread reference — middle reads
~0° by construction. Mapped via `map_abduction_to_motor` (signed, unlike `map_to_motor`),
since spreading toward the thumb side vs. the pinky side are opposite directions, not a
single open→closed range.

### 4.3 Live per-joint calibration

`map_to_motor`'s `(open_deg, closed_deg)` window isn't the same for every joint — e.g. the
thumb's CMC/MCP joints have a visibly smaller raw-angle swing than finger PIP/DIP joints,
so a single generic window can leave a joint stuck near one end regardless of pose. Rather
than hand-tune per-joint constants: **click the "Hand Tracking" video window** (it must
have keyboard focus, not your terminal), hold your hand fully open and press **`o`**, then
make a fist and press **`c`**. Each press reads back that frame's per-joint angles and
updates `joint_bounds` for whichever joints you were actually showing the camera at that
moment; a "Captured OPEN/CLOSED bounds" table prints to confirm. Falls back to
`DEFAULT_OPEN_DEG=160, DEFAULT_CLOSED_DEG=90` until you calibrate. Not persisted across
runs — recalibrate each session, or hardcode your values into `joint_bounds`'s
initialization if they're stable for you.

### 4.4 Tuning knobs

| Constant | Effect |
|---|---|
| `FILTER_PARAMS[name]` (`mincutoff`, `beta`, `dcutoff`) | Per-finger OneEuroFilter tuning. Lower `mincutoff` = more smoothing at rest (fixes jitter); higher `beta` = less lag during fast motion. Thumb already uses more aggressive values than the others — retune similarly if a specific finger is noisy. |
| `joint_bounds` | Per-joint `(open_deg, closed_deg)` — set via live calibration (§4.3), not hand-edited normally. |
| `map_abduction_to_motor`'s `max_input_deg`/`max_motor_deg` | Spread sensitivity — how many degrees of real spread maps to the full ±25° motor-degree output. |
| `PRINT_EVERY_N_FRAMES` | Console status print throttle. |

---

## 5. `leap_hand_walk.py` — experimental walking gait

Scripted (not learned) gait: Index/Middle/Ring act as three legs, Thumb is a fixed
support point. No existing project walks a LEAP Hand specifically; the closest references
are ETH Zurich's ["Fingers as Legs"](https://arxiv.org/abs/2609.17172) paper (RL-trained,
needed because its 5 fingers are mechanically unequal) and the older DLR-Crawler (scripted
tripod gait on 3-fingered hands). LEAP's Index/Middle/Ring are mechanically identical
(same links/limits, ~45mm apart per `leap-hand/LEAP_Hand_Sim`'s URDF), which is what makes
a classical scripted gait — no simulator/training needed — a reasonable starting point
here.

### 5.1 Leg model

Per leg, two joints do the work; MCP_Side is held neutral (no splay) and DIP is held at a
fixed curl ("ankle") — both simplifications, not fundamental limits:

- **MCP_Forward ("hip")**: sweeps the whole finger fore/aft. During **stance** (long
  phase) it sweeps slowly — this is what drags the body forward over the planted
  fingertip. During **swing** (short phase) it resets quickly back to the start.
  Direction matters and was **determined empirically, not from theory**: the original
  "retract-during-stance" direction (modeled after a normal leg's stance-phase
  retraction) actually drove the hand *backward* on real hardware; the sweep direction
  in the code now is the one confirmed to go forward. Use `--reverse` to flip it back
  for intentional backward motion.
- **PIP ("knee")**: extended (foot down) during stance, flexes briefly during swing so
  the fingertip clears the ground while the hip resets.

`leg_trajectory_deg(phase, reverse)` implements one leg's cycle: a cosine ease
(`_ease`, zero velocity at phase boundaries so legs don't jerk) drives the hip smoothly
between `HIP_MIN_DEG`/`HIP_MAX_DEG`, and a sine bump drives the knee between
`KNEE_STANCE_DEG`/`KNEE_SWING_DEG` during the swing window only.

### 5.2 Gait patterns

`build_pose_deg(global_phase, synchronized, reverse)` assembles the full 16-element pose
by giving each leg its own phase:

- **Wave gait** (`synchronized=False`, default): legs phase-offset by 1/3 of a cycle each,
  so with `DUTY_FACTOR=2/3` exactly one leg is ever in swing — the other two (plus the
  thumb) are always planted. Maximizes stability, but only ever has at most 2 legs
  actively pushing at once.
- **Synchronized gait** (`synchronized=True`, `--sync`): all three legs move in lockstep —
  stance together (3× the simultaneous propulsive force) and swing together (hand briefly
  rests on just the thumb). Confirmed on hardware to produce more reliable forward motion
  once the direction/stride issues below were fixed. Less stable during the shared swing
  moment.

### 5.3 Thumb support pose

`THUMB_POSE_DEG` is a fixed 4-element pose (Side, Forward, PIP, DIP) held throughout the
gait — not derived from geometry, **captured empirically**: `--mode thumb-capture`
disables torque on just the thumb's 4 motors (after a brief ramp so nothing jumps) so you
can pose it by hand, then prints its actual joint angles once a second so you can read off
a good value and hardcode it into `THUMB_POSE_DEG`. Always sanity-check a newly captured
value against `lhu.LEAPsim_limits()` before using it — captured values have come back
within a degree of a hard joint limit more than once this session; back off ~10° from any
margin under ~10-15° so the safety clip doesn't pin the joint against its mechanical stop
under continuous position-current control. `--mode thumb-pose` holds the current
`THUMB_POSE_DEG` (legs planted/static) for visually checking a value without the legs
moving.

### 5.4 Running it

```bash
# Safest first step: cycle one finger only, other two held planted, thumb fixed.
python leap_hand_walk.py --mode single-leg --finger Index --cycle-period 3.0 --cycles 4

# Full gait once single-leg checks out for all three fingers:
python leap_hand_walk.py --mode walk --cycle-period 3.0 --cycles 4
python leap_hand_walk.py --mode walk --sync --cycle-period 3.0 --cycles 4   # synchronized
python leap_hand_walk.py --mode walk --reverse --cycle-period 3.0 --cycles 4 # backward

# Thumb pose tools:
python leap_hand_walk.py --mode thumb-capture --duration 30   # pose it by hand, read values
python leap_hand_walk.py --mode thumb-pose --duration 5       # hold current THUMB_POSE_DEG

--cycles 0   # any walk/single-leg mode: run until Ctrl-C instead of a fixed count
```

### 5.5 Safety mechanisms (don't remove without replacing)

- **`_to_real_radians()`** always runs every target pose through `lhu.angle_safety_clip`
  before it's sent to the motors — this is what prevents a bad constant from commanding
  past a joint's hard limit (it'll just pin at the limit instead, which is still not
  great under sustained load — see §5.3 — but won't exceed it).
- **`slew_limit()`** (shared with `hand_demo.py`) caps commanded-position rate of change
  as a backstop, even though the trajectories themselves are already smooth by
  construction.
- **SIGTERM handler**: converts `SIGTERM` into a `KeyboardInterrupt` so `finally` blocks
  (specifically, re-enabling thumb torque in `capture_thumb_pose`) still run if the
  process is killed externally (e.g. by a `timeout` wrapper) instead of via Ctrl-C. This
  was added after a real incident where a `timeout`-killed capture session left two thumb
  motors permanently torque-disabled. If you ever suspect this happened (motor won't hold
  position, feels limp), check and fix directly:
  ```python
  import dynamixel_sdk as dxl
  port = dxl.PortHandler('/dev/ttyUSB0'); port.openPort(); port.setBaudRate(4000000)
  packet = dxl.PacketHandler(2.0)
  for motor_id in [12, 13, 14, 15]:
      packet.write1ByteTxRx(port, motor_id, 64, 1)  # ADDR_TORQUE_ENABLE=64, enable=1
  ```

### 5.6 Tuning knobs

| Constant | Effect | Notes |
|---|---|---|
| `HIP_MIN_DEG` / `HIP_MAX_DEG` | Stride length (hip sweep range) | Real range is −18°/+128° for Index/Middle/Ring's MCP_Forward. Started at 15°/65° (50° sweep) which produced visible stepping but **zero net translation**; widened to −5°/110° (115° sweep) fixed it. Keep ≥10-15° margin from the hard limits. |
| `KNEE_STANCE_DEG` / `KNEE_SWING_DEG` | Ground clearance during swing | Real PIP range is −29°/+108°. |
| `ANKLE_DEG` | Fixed DIP curl | Not actively cycled in this version. |
| `DUTY_FACTOR` | Fraction of cycle in stance vs. swing | `2/3` is what guarantees exactly one wave-gait leg swings at a time; changing it changes that guarantee. |
| `THUMB_POSE_DEG` | Thumb support pose | Capture via `--mode thumb-capture`, don't hand-guess (§5.3). |
| `MAX_SLEW_DEG_PER_S` | Safety backstop, not a speed control | Also sets a **speed ceiling**: below a cycle period of roughly `(KNEE_SWING_DEG - KNEE_STANCE_DEG) * π / MAX_SLEW_DEG_PER_S / (1 - DUTY_FACTOR)` seconds, the knee's swing-phase lift needs more angular velocity than this cap allows and gets clipped/distorted rather than just "faster." At the current constants that threshold is ≈1.06s/cycle. Raising this constant trades away margin against a bug producing a runaway jump — only do it deliberately (we tested up to ~460°/s, just under the motor's real 486°/s hardware limit per the URDF's `velocity="8.48"` rad/s, and decided against it for the margin loss). |
| `--cycle-period` (CLI) | Overall gait speed | 3.0s = validated safe/stable baseline. 1.1s is the fastest tested that stays under the `MAX_SLEW_DEG_PER_S` cap. |

### 5.7 Known limitations / open questions

- Direction of net travel and its magnitude haven't been fully characterized across
  cycle-period/duty-factor/stride combinations — only a handful of configurations have
  been tested on hardware.
- The palm's underside may still drag on the ground for some fraction of the cycle
  (confirmed via side-camera footage at the old, shorter stride — not re-verified since
  the stride widening).
- Thumb is a fixed prop, not an active leg — it isn't synchronized into the gait cycle at
  all, so as the body translates it's effectively dragging/re-anchoring passively rather
  than stepping.
- No steering: the gait only goes straight forward or straight backward.
