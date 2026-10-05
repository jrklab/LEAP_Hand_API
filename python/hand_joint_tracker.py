import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import socket

# === UDP Setup ===
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class EMA:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

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
ema_buffers = {
    name: [EMA(alpha=0.05) for _ in range(3)]
    for name in fingers
}
# Spread/abduction smoother buffers -- Index/Middle/Ring only (Thumb's abduction is
# mechanically distinct -- opposition across the palm, not spread between fingers -- and
# isn't modeled here; its motor side-joint is left at the existing neutral default).
ABDUCTION_FINGERS = ['Index', 'Middle', 'Ring']
abduction_ema = {name: EMA(alpha=0.05) for name in ABDUCTION_FINGERS}

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
        print("\n--- Finger Angles ---")
        smoothed_angles = []

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

                mcp_angle = ema_buffers[name][0].update(get_angle(wrist, mcp, pip))
                pip_angle = ema_buffers[name][1].update(get_angle(mcp, pip, dip))
                dip_angle = ema_buffers[name][2].update(get_angle(pip, dip, tip))

                mcp_mapped = int(map_to_motor(mcp_angle))
                pip_mapped = int(map_to_motor(pip_angle))
                dip_mapped = int(map_to_motor(dip_angle))

                smoothed_angles.extend([mcp_mapped, pip_mapped, dip_mapped])

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
                smoothed = abduction_ema[name].update(raw_angle)
                abduction_angles.append(int(map_abduction_to_motor(smoothed)))

            print(f"Abduction: Index={abduction_angles[0]} Middle={abduction_angles[1]} Ring={abduction_angles[2]}")
            smoothed_angles.extend(abduction_angles)
        except Exception as e:
            print(f"Abduction estimation failed: {e}")
            smoothed_angles.extend([0, 0, 0])

        # === Send via UDP
        packet = ",".join(str(angle) for angle in smoothed_angles)
        sock.sendto(packet.encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
