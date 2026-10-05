#!/usr/bin/env python3
"""
Experimental scripted walking gait for the LEAP Hand, using Index/Middle/Ring as three
"legs" and the Thumb as a fixed fourth support point ("kickstand") -- not a learned
policy, just classical joint-space trajectory generation. This is a reasonable starting
point because Index/Middle/Ring are mechanically identical (same links, same joint
limits -- see leap-hand/LEAP_Hand_Sim's robot.urdf) and evenly spaced ~45mm apart in a
row, unlike e.g. a human-proportioned 5-finger hand (ETH's "Fingers as Legs" paper, whose
unequal fingers needed an RL policy to coordinate).

Per leg, two joints do the work (all angles in the "allegro" convention already used
throughout this repo: 0 roughly "neutral/open", positive = curling further closed -- see
leap_hand_utils.py):
  - MCP_Forward ("hip"): sweeps the whole finger fore/aft. During stance this sweeps
    slowly from "curled back" to "reaching forward", dragging the body forward over the
    planted fingertip. (The opposite sweep direction was tried first by analogy to a
    normal leg's stance-phase retraction, but empirically drove the hand backward --
    reversed here based on that result, not re-derived from theory.) During swing it
    resets quickly back to "curled back".
  - PIP ("knee"): held extended (foot down) during stance, lifts briefly during swing so
    the fingertip clears the ground while the hip resets.
MCP_Side is held neutral (no splay) and DIP is held at a fixed curl ("ankle") -- both
simplifications for this first version, not because they can't help later.

Gait: a 3-leg wave gait (duty factor 2/3) -- Index, Middle, Ring are phase-offset by 1/3
of a cycle each, so exactly one leg swings at a time and the other two (plus the fixed
thumb) always support the hand. Same statically-stable principle as a hexapod wave gait,
just with 3 legs instead of 6 (c.f. DLR-Crawler's tripod gait, which was scripted rather
than learned).

This is a first-pass, physically-grounded guess, not a validated gait -- amplitude, duty
factor, phase order, thumb pose, and surface friction will likely all need retuning once
you see how the real hardware actually moves (both reference projects needed real-world
tuning/training too). Start with --mode single-leg (and ideally prop/hold the hand
securely) to sanity check one leg's motion before trying the full gait.
"""
import argparse
import math
import signal
import time

import numpy as np

import leap_hand_utils.leap_hand_utils as lhu
from hand_demo import LeapNode, slew_limit

LEG_FINGER_SLICES = {
    'Index': slice(0, 4),
    'Middle': slice(4, 8),
    'Ring': slice(8, 12),
}
LEG_ORDER = ['Index', 'Middle', 'Ring']  # wave-gait phase offset follows this order

# All angles below are in the "allegro" convention already used throughout this repo:
# 0 roughly "neutral/open", positive = curling further closed. See leap_hand_utils.py.
# Camera footage showed real lift-off but zero net body translation cycle-over-cycle --
# the hip sweep (50deg) was likely too short to generate meaningful stride length.
# MCP_Forward's real range is -18/+128deg; widened the sweep to use most of it, keeping
# ~13-18deg margin on each end.
HIP_MIN_DEG = -5.0      # fully "reaching forward" (start of stance / end of swing)
HIP_MAX_DEG = 110.0     # fully "curled back" (end of stance / start of swing)
KNEE_STANCE_DEG = 10.0  # extended, foot planted
KNEE_SWING_DEG = 55.0   # flexed, foot lifted
ANKLE_DEG = 20.0        # fixed DIP curl, not actively cycled
# Fixed "kickstand" pose: Side,Forward,PIP,DIP. Captured by disabling thumb torque
# (--mode thumb-capture) and manually posing the thumb for ground support, then reading
# back its settled joint angles -- not a geometric guess.
THUMB_POSE_DEG = [99.8, -7.4, 8.3, -5.5]

DUTY_FACTOR = 2.0 / 3.0  # fraction of each leg's cycle spent in stance
MAX_SLEW_DEG_PER_S = 400.0  # safety backstop only -- the trajectory itself is already smooth


def _ease(s):
    """Cosine ease-in/out: zero velocity at s=0 and s=1, so a leg doesn't jerk at the
    stance/swing phase boundary."""
    return (1 - math.cos(math.pi * s)) / 2


def leg_trajectory_deg(phase, reverse=False):
    """phase in [0, 1), wraps automatically. Returns (hip_deg, knee_deg) for one leg at
    this point in its own gait cycle.

    reverse=False (default, drives the hand forward): stance sweeps MAX->MIN, swing
    resets MIN->MAX -- empirically confirmed direction, not derived from theory (the
    opposite convention was tried first and drove the hand backward).
    reverse=True: stance sweeps MIN->MAX instead, for backward motion."""
    hip_stance_start, hip_stance_end = (HIP_MIN_DEG, HIP_MAX_DEG) if reverse else (HIP_MAX_DEG, HIP_MIN_DEG)
    phase = phase % 1.0
    if phase < DUTY_FACTOR:
        s = phase / DUTY_FACTOR
        hip = hip_stance_start + _ease(s) * (hip_stance_end - hip_stance_start)
        knee = KNEE_STANCE_DEG
    else:
        s = (phase - DUTY_FACTOR) / (1 - DUTY_FACTOR)
        hip = hip_stance_end + _ease(s) * (hip_stance_start - hip_stance_end)
        knee = KNEE_STANCE_DEG + (KNEE_SWING_DEG - KNEE_STANCE_DEG) * math.sin(math.pi * s)
    return hip, knee


def build_pose_deg(global_phase, synchronized=False, reverse=False):
    """Full 16-element allegro-convention pose (degrees) for one instant of the gait.

    synchronized=False (default): wave gait -- legs phase-offset by 1/3 of a cycle, so
    exactly one leg swings at a time and the hand always has >=2 feet (+thumb) down.
    synchronized=True: all three legs move in lockstep -- stance together (3x the
    simultaneous propulsive force of the wave gait's at-most-2-legs-pushing) and swing
    together (briefly resting on just the thumb while all three reset). Worth trying if
    the wave gait's staggered, smaller per-instant thrust isn't enough to overcome
    whatever's resisting the palm, at the cost of a less stable swing moment.
    """
    pose = np.zeros(16)
    pose[12:16] = THUMB_POSE_DEG
    for i, name in enumerate(LEG_ORDER):
        leg_phase = global_phase if synchronized else global_phase + i / len(LEG_ORDER)
        hip_deg, knee_deg = leg_trajectory_deg(leg_phase, reverse=reverse)
        pose[LEG_FINGER_SLICES[name]] = [0.0, hip_deg, knee_deg, ANKLE_DEG]
    return pose


def _to_real_radians(pose_deg):
    """allegro-convention degrees -> safety-clipped real LEAPhand radians (the space
    write_desired_pos/set_leap expect)."""
    return lhu.angle_safety_clip(lhu.allegro_to_LEAPhand(np.deg2rad(pose_deg), zeros=False))


def run(leap_hand, cycle_period_s, num_cycles, control_hz=50, synchronized=False, reverse=False):
    control_period_s = 1.0 / control_hz
    max_step = np.deg2rad(MAX_SLEW_DEG_PER_S) * control_period_s
    pos = leap_hand.curr_pos.copy()
    t0 = time.time()
    duration_s = cycle_period_s * num_cycles if num_cycles else None
    while duration_s is None or time.time() - t0 < duration_s:
        global_phase = ((time.time() - t0) / cycle_period_s) % 1.0
        target = _to_real_radians(build_pose_deg(global_phase, synchronized=synchronized, reverse=reverse))
        pos = slew_limit(pos, target, max_step)
        leap_hand.set_leap(pos)
        time.sleep(control_period_s)


def run_single_leg(leap_hand, finger, cycle_period_s, num_cycles, control_hz=50):
    """Cycle only one leg through its gait trajectory, the other two legs held planted
    forward (stance pose) and the thumb at its fixed kickstand pose -- use this first to
    sanity check one leg's motion (ideally with the hand propped/held securely) before
    trying the full 3-leg gait."""
    control_period_s = 1.0 / control_hz
    max_step = np.deg2rad(MAX_SLEW_DEG_PER_S) * control_period_s
    pos = leap_hand.curr_pos.copy()
    t0 = time.time()
    duration_s = cycle_period_s * num_cycles if num_cycles else None
    while duration_s is None or time.time() - t0 < duration_s:
        phase = ((time.time() - t0) / cycle_period_s) % 1.0
        pose_deg = np.zeros(16)
        pose_deg[12:16] = THUMB_POSE_DEG
        for name in LEG_ORDER:
            if name == finger:
                hip_deg, knee_deg = leg_trajectory_deg(phase)
            else:
                hip_deg, knee_deg = HIP_MIN_DEG, KNEE_STANCE_DEG  # held planted/forward
            pose_deg[LEG_FINGER_SLICES[name]] = [0.0, hip_deg, knee_deg, ANKLE_DEG]
        target = _to_real_radians(pose_deg)
        pos = slew_limit(pos, target, max_step)
        leap_hand.set_leap(pos)
        time.sleep(control_period_s)


def hold_thumb_pose(leap_hand, duration_s, control_hz=50):
    """Hold THUMB_POSE_DEG with the three legs planted forward/static -- for eyeballing
    and retuning the thumb's static support pose in isolation, without the legs moving."""
    control_period_s = 1.0 / control_hz
    max_step = np.deg2rad(MAX_SLEW_DEG_PER_S) * control_period_s
    pos = leap_hand.curr_pos.copy()
    pose_deg = np.zeros(16)
    pose_deg[12:16] = THUMB_POSE_DEG
    for name in LEG_ORDER:
        pose_deg[LEG_FINGER_SLICES[name]] = [0.0, HIP_MIN_DEG, KNEE_STANCE_DEG, ANKLE_DEG]
    target = _to_real_radians(pose_deg)
    t0 = time.time()
    while time.time() - t0 < duration_s:
        pos = slew_limit(pos, target, max_step)
        leap_hand.set_leap(pos)
        time.sleep(control_period_s)


THUMB_MOTORS = [12, 13, 14, 15]


def capture_thumb_pose(leap_hand, duration_s, control_hz=20):
    """Hold Index/Middle/Ring in their stance pose (torque on, so the hand stays
    supported) and disable torque on just the thumb so it can be moved by hand into a
    support position. Prints the thumb's actual joint angles (Side, Forward, PIP, DIP, in
    the same allegro-convention degrees THUMB_POSE_DEG uses) once a second so a good
    manually-found pose can be read off and copied back into THUMB_POSE_DEG."""
    control_period_s = 1.0 / control_hz
    max_step = np.deg2rad(MAX_SLEW_DEG_PER_S) * control_period_s
    pos = leap_hand.curr_pos.copy()
    pose_deg = np.zeros(16)
    pose_deg[12:16] = THUMB_POSE_DEG  # only matters until torque is disabled below
    for name in LEG_ORDER:
        pose_deg[LEG_FINGER_SLICES[name]] = [0.0, HIP_MIN_DEG, KNEE_STANCE_DEG, ANKLE_DEG]
    target = _to_real_radians(pose_deg)

    # Ramp into the stance pose first, while the thumb still has torque (avoids a sudden
    # jump the instant torque is cut).
    t0 = time.time()
    while time.time() - t0 < 1.5:
        pos = slew_limit(pos, target, max_step)
        leap_hand.set_leap(pos)
        time.sleep(control_period_s)

    leap_hand.dxl_client.set_torque_enabled(THUMB_MOTORS, False)
    print("Thumb torque disabled -- move it by hand into a supporting position.")
    print("Printing thumb joint angles (Side, Forward, PIP, DIP) in degrees every second (Ctrl-C to stop)...")
    try:
        last_print = 0.0
        t0 = time.time()
        while duration_s <= 0 or time.time() - t0 < duration_s:
            pos[0:12] = target[0:12]  # keep Index/Middle/Ring held in stance
            leap_hand.set_leap(pos)
            now = time.time()
            if now - last_print >= 1.0:
                current_real = leap_hand.read_pos()
                current_allegro_deg = np.degrees(lhu.LEAPhand_to_allegro(current_real, zeros=False))
                thumb_deg = current_allegro_deg[12:16]
                print(f"Thumb pose (deg): Side={thumb_deg[0]:.1f}  Forward={thumb_deg[1]:.1f}  "
                      f"PIP={thumb_deg[2]:.1f}  DIP={thumb_deg[3]:.1f}")
                last_print = now
            time.sleep(control_period_s)
    finally:
        leap_hand.dxl_client.set_torque_enabled(THUMB_MOTORS, True)
        print("Thumb torque re-enabled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["single-leg", "walk", "thumb-pose", "thumb-capture"], default="single-leg",
                        help="'single-leg' cycles one finger only (safer first test); 'walk' runs the full "
                             "3-leg wave gait; 'thumb-pose' holds THUMB_POSE_DEG still for visual tuning; "
                             "'thumb-capture' disables thumb torque so it can be moved by hand, and prints its "
                             "angles so a manually-found pose can be read off.")
    parser.add_argument("--finger", choices=LEG_ORDER, default="Index",
                        help="Which finger to cycle in --mode single-leg.")
    parser.add_argument("--cycle-period", type=float, default=3.0,
                        help="Seconds per full gait cycle (slower = safer to watch).")
    parser.add_argument("--cycles", type=int, default=4, help="Number of cycles to run, 0 = run until Ctrl-C.")
    parser.add_argument("--sync", action="store_true",
                        help="In --mode walk, move all three legs in lockstep (stance/swing together) instead "
                             "of the default staggered wave gait -- trades continuous ground support for 3x "
                             "the simultaneous propulsive force.")
    parser.add_argument("--reverse", action="store_true",
                        help="In --mode walk, drive the hand backward instead of forward.")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds to hold the pose in --mode thumb-pose, or to print in --mode thumb-capture "
                             "(0 = until Ctrl-C).")
    args = parser.parse_args()

    # Without this, an external kill (e.g. a `timeout` wrapper, as happened once while
    # testing thumb-capture) sends SIGTERM, whose default disposition terminates the
    # process immediately -- skipping every `finally` block, including the one that
    # re-enables thumb torque. Converting it to a KeyboardInterrupt lets normal Python
    # exception unwinding (and the try/except below) run that cleanup regardless of how
    # the process is asked to stop.
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))

    leap_hand = LeapNode(motors=list(range(16)))
    try:
        if args.mode == "single-leg":
            run_single_leg(leap_hand, args.finger, args.cycle_period, args.cycles)
        elif args.mode == "thumb-pose":
            hold_thumb_pose(leap_hand, args.duration)
        elif args.mode == "thumb-capture":
            capture_thumb_pose(leap_hand, args.duration)
        else:
            run(leap_hand, args.cycle_period, args.cycles, synchronized=args.sync, reverse=args.reverse)
    except KeyboardInterrupt:
        pass
