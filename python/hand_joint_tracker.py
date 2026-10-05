import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import socket
import time

# === UDP Setup ===
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al. 2012): applies heavy smoothing when the
    signal is nearly still (killing jitter) and light smoothing when it's moving fast
    (killing lag), instead of a fixed-alpha EMA's constant lag/noise tradeoff regardless of
    speed -- useful here since a fast gesture and a held pose have very different noise
    characteristics from the same landmark estimator.
    """
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def update(self, x, timestamp=None):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            return x
        if timestamp is not None and self.t_prev is not None and timestamp > self.t_prev:
            self.freq = 1.0 / (timestamp - self.t_prev)
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat

# === Angle Utilities ===
def get_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def map_to_motor(angle, open_deg=160, closed_deg=90, max_motor_deg=90):
    angle = np.clip(angle, closed_deg, open_deg)
    return (open_deg - angle) / (open_deg - closed_deg) * max_motor_deg

def map_abduction_to_motor(angle_deg, max_input_deg=25, max_motor_deg=25):
    """Signed linear map for finger spread (abduction/adduction), unlike map_to_motor's
    one-sided open/closed clip -- spreading toward the thumb side vs. the pinky side are
    opposite directions from the middle finger (the spread reference), not a single
    open->closed range."""
    angle_deg = np.clip(angle_deg, -max_input_deg, max_input_deg)
    return (angle_deg / max_input_deg) * max_motor_deg

def to_np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

def signed_abduction_angle(ref_vec, finger_vec, normal):
    """Angle (degrees, signed) of finger_vec relative to ref_vec, both projected onto the
    palm plane defined by `normal`. Sign follows the right-hand rule around `normal`, so
    fingers spreading to opposite sides of the reference naturally come out with opposite
    signs -- no per-finger sign table needed."""
    ref_proj = ref_vec - np.dot(ref_vec, normal) * normal
    finger_proj = finger_vec - np.dot(finger_vec, normal) * normal
    ref_norm = np.linalg.norm(ref_proj)
    finger_norm = np.linalg.norm(finger_proj)
    if ref_norm < 1e-8 or finger_norm < 1e-8:
        return 0.0
    ref_proj /= ref_norm
    finger_proj /= finger_norm
    cos_angle = np.clip(np.dot(ref_proj, finger_proj), -1.0, 1.0)
    sin_angle = np.dot(normal, np.cross(ref_proj, finger_proj))
    return np.degrees(np.arctan2(sin_angle, cos_angle))

def average_landmarks(landmarks, indices):
    coords = np.mean([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices], axis=0)
    palm_center = landmark_pb2.NormalizedLandmark()
    palm_center.x, palm_center.y, palm_center.z = coords
    return palm_center

# === MediaPipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# === Finger Definitions ===
fingers = {
    'Thumb': [1, 2, 3, 4],       # CMC, MCP, IP, TIP
    'Index': [5, 6, 7, 8],       # MCP, PIP, DIP, TIP
    'Middle': [9, 10, 11, 12],
    'Ring': [13, 14, 15, 16]
}
WRIST = 0

# === Smoother Buffers: 4 fingers × 3 joints
# mincutoff/beta are starting points (same shape as the One Euro paper's own defaults) --
# raise beta if fast gestures still feel laggy, lower mincutoff if a held pose still jitters.
#
# Thumb gets its own, more aggressive, settings: its "MCP" angle is computed using the
# wrist as a stand-in vertex (the other fingers use a real adjacent joint), and the thumb
# moves more through the camera's depth (z) axis, which MediaPipe tracks less reliably than
# x/y -- both make its raw angle noisier than the other fingers', so it's smoothed harder
# (lower mincutoff/beta/dcutoff) at the cost of a bit more lag.
FILTER_PARAMS = {
    'Thumb':  {'mincutoff': 0.3, 'beta': 0.1, 'dcutoff': 0.5},
    'Index':  {'mincutoff': 1.0, 'beta': 0.3, 'dcutoff': 1.0},
    'Middle': {'mincutoff': 1.0, 'beta': 0.3, 'dcutoff': 1.0},
    'Ring':   {'mincutoff': 1.0, 'beta': 0.3, 'dcutoff': 1.0},
}
joint_filters = {
    name: [OneEuroFilter(**FILTER_PARAMS[name]) for _ in range(3)]
    for name in fingers
}
# Spread/abduction smoother buffers -- Index/Middle/Ring only (Thumb's abduction is
# mechanically distinct -- opposition across the palm, not spread between fingers -- and
# isn't modeled here; its motor side-joint is left at the existing neutral default).
ABDUCTION_FINGERS = ['Index', 'Middle', 'Ring']
abduction_filters = {name: OneEuroFilter(mincutoff=1.0, beta=0.3) for name in ABDUCTION_FINGERS}

# Per-joint (open_deg, closed_deg) bounds for map_to_motor. A single generic (160, 90)
# window doesn't fit every joint -- e.g. the thumb's CMC/MCP joints have a visibly smaller
# raw-angle swing than finger PIP/DIP joints, so with the generic window they may never
# read low enough to leave the "open" end of the motor range regardless of pose, while the
# joint whose real range happens to overlap the window moves normally. Rather than guess
# per-joint numbers, capture them live: hold the hand fully OPEN and press 'o', then make a
# FIST and press 'c' -- this records each joint's own true extremes for this hand/session.
# Defaults below are only a fallback until you calibrate.
DEFAULT_OPEN_DEG = 160.0
DEFAULT_CLOSED_DEG = 90.0
joint_bounds = {
    name: [[DEFAULT_OPEN_DEG, DEFAULT_CLOSED_DEG] for _ in range(3)]
    for name in fingers
}
latest_raw_angles = {name: [None, None, None] for name in fingers}
FINGER_ORDER = ['Thumb', 'Index', 'Middle', 'Ring']

def print_joint_bounds(bounds):
    print("--- Calibrated (open_deg, closed_deg) per joint ---")
    for name in FINGER_ORDER:
        mcp, pip, dip = bounds[name]
        print(
            f"  {name:6s}: MCP=({mcp[0]:.1f},{mcp[1]:.1f})  "
            f"PIP=({pip[0]:.1f},{pip[1]:.1f})  DIP=({dip[0]:.1f},{dip[1]:.1f})"
        )

print(
    "Hold the hand fully OPEN and press 'o', then make a FIST and press 'c', to calibrate "
    "per-joint angle ranges (defaults used until then). Press ESC to quit."
)

# Status prints below are throttled to this many frames (~camera fps) to stay readable.
PRINT_EVERY_N_FRAMES = 15
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        frame_count += 1
        verbose = (frame_count % PRINT_EVERY_N_FRAMES == 0)
        if verbose:
            print("\n--- Finger Angles ---")
        smoothed_angles = []
        now = time.time()  # shared across every joint filter this frame

        for name, ids in fingers.items():
            try:
                wrist = landmarks[WRIST]
                # palm_center = average_landmarks(landmarks, [0, 5, 9, 13, 17])
                # print palm_center and wrist
                # print(f"{name} Palm Center: {palm_center.x:.2f}, {palm_center.y:.2f}, {palm_center.z:.2f}")
                # print(f"{name} Wrist: {wrist.x:.2f}, {wrist.y:.2f}, {wrist.z:.2f}")
                mcp = landmarks[ids[0]]
                pip = landmarks[ids[1]]
                dip = landmarks[ids[2]]
                tip = landmarks[ids[3]]

                mcp_angle = joint_filters[name][0].update(get_angle(wrist, mcp, pip), now)
                pip_angle = joint_filters[name][1].update(get_angle(mcp, pip, dip), now)
                dip_angle = joint_filters[name][2].update(get_angle(pip, dip, tip), now)
                latest_raw_angles[name] = [mcp_angle, pip_angle, dip_angle]

                (mcp_open, mcp_closed), (pip_open, pip_closed), (dip_open, dip_closed) = joint_bounds[name]
                mcp_mapped = int(map_to_motor(mcp_angle, open_deg=mcp_open, closed_deg=mcp_closed))
                pip_mapped = int(map_to_motor(pip_angle, open_deg=pip_open, closed_deg=pip_closed))
                dip_mapped = int(map_to_motor(dip_angle, open_deg=dip_open, closed_deg=dip_closed))

                smoothed_angles.extend([mcp_mapped, pip_mapped, dip_mapped])

                if verbose:
                    print(f"{name}: MCP={mcp_mapped}°, PIP={pip_mapped}°, DIP={dip_mapped}°")

            except Exception as e:
                print(f"{name}: angle estimation failed: {e}")

        # === Spread/abduction (MCP side-to-side) for Index/Middle/Ring ===
        # Palm plane normal from wrist + index/ring MCPs (the three widest-spaced, most
        # stable palm points we track); middle finger's MCP->PIP direction, projected onto
        # that plane, is the zero-spread reference -- it reads ~0 deg by construction.
        try:
            wrist = to_np(landmarks[WRIST])
            index_mcp, index_pip = to_np(landmarks[5]), to_np(landmarks[6])
            middle_mcp, middle_pip = to_np(landmarks[9]), to_np(landmarks[10])
            ring_mcp, ring_pip = to_np(landmarks[13]), to_np(landmarks[14])

            normal = np.cross(index_mcp - wrist, ring_mcp - wrist)
            normal /= np.linalg.norm(normal)
            ref_vec = middle_pip - middle_mcp

            finger_vecs = {
                'Index': index_pip - index_mcp,
                'Middle': ref_vec,
                'Ring': ring_pip - ring_mcp,
            }
            abduction_angles = []
            for name in ABDUCTION_FINGERS:
                raw_angle = signed_abduction_angle(ref_vec, finger_vecs[name], normal)
                smoothed = abduction_filters[name].update(raw_angle, now)
                abduction_angles.append(int(map_abduction_to_motor(smoothed)))

            if verbose:
                print(f"Abduction: Index={abduction_angles[0]} Middle={abduction_angles[1]} Ring={abduction_angles[2]}")
            smoothed_angles.extend(abduction_angles)
        except Exception as e:
            print(f"Abduction estimation failed: {e}")
            smoothed_angles.extend([0, 0, 0])

        # === Send via UDP
        packet = ",".join(str(angle) for angle in smoothed_angles)
        sock.sendto(packet.encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("Hand Tracking", image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('o') or key == ord('c'):
        bound_index = 0 if key == ord('o') else 1  # [open, closed]
        label = "OPEN" if key == ord('o') else "CLOSED"
        if all(latest_raw_angles[name][j] is None for name in fingers for j in range(3)):
            print(f"No hand detected -- '{label}' capture ignored, show your hand to the camera first.")
        else:
            for name in fingers:
                for j in range(3):
                    angle = latest_raw_angles[name][j]
                    if angle is not None:
                        joint_bounds[name][j][bound_index] = angle
            # Guard against a joint whose two captured poses came out reversed (e.g. 'o' and
            # 'c' pressed in the wrong order) -- map_to_motor assumes open_deg >= closed_deg.
            for name in fingers:
                for j in range(3):
                    open_deg, closed_deg = joint_bounds[name][j]
                    if open_deg < closed_deg:
                        joint_bounds[name][j] = [closed_deg, open_deg]
            print(f"\n*** Captured {label} pose ***")
            print_joint_bounds(joint_bounds)

cap.release()
cv2.destroyAllWindows()
