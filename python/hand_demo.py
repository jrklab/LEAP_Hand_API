import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
import socket
import threading
#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more in radians.

#LEAP hand conventions:
#3.14 rad is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed in radians.
#The MCP is centered at 3.14 and can move positive or negative to that in radians.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

"""
########################################################

def slew_limit(current, target, max_step):
    """Clamp how far `current` can move toward `target` in one control step, so a single
    noisy/garbled UDP packet (or a vision-side glitch) can't produce an instantaneous large
    jump in commanded position -- same idea as the 32-step init ramp in LeapNode.__init__,
    applied continuously instead of once at startup."""
    delta = np.clip(target - current, -max_step, max_step)
    return current + delta

class LeapNode:
    def __init__(self, motors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350  ##set this to 550 if you are using full motors!!!!
        init_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        init_steps = 32
        step_interval = 0.03
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        ## self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.motors = motors
        self.prev_pos = self.pos = self.curr_pos = init_pos[self.motors]
        print(f"Using motors ID: {self.motors}")
        baud_rate = 4000000 # default baud rate was 4000000, but 1000000 is more stable for my system
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', baud_rate)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', baud_rate)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', baud_rate)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True, retries=3)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.curr_pos = self.read_pos()
        # print current position
        print(f"Initial Position: {self.curr_pos}")
        angle_per_step = (init_pos[self.motors] - self.curr_pos)/init_steps
        for i in range(init_steps):
            self.curr_pos += angle_per_step
            self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
            time.sleep(step_interval)
        # self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose[self.motors])
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()

    def recover_stalled_motors(self, motor_ids, reboot_if_needed=True):
        """Check the given motors for torque that the firmware auto-disabled on its own
        (e.g. an Overload trip from a mechanical clash against another finger) and try to
        recover them, instead of leaving that motor dead until the whole script restarts.

        First tries a plain re-enable (works if the physical clash has already cleared --
        many firmware versions drop the latched error once the triggering condition is
        gone and torque is requested again). If that doesn't take, reboots just the
        still-stalled motors and re-applies this node's PID gains/current limit to them
        (reboot resets a motor's RAM-area settings to firmware defaults, including those).
        Returns the list of motor IDs that were found stalled (recovered or not)."""
        torque_state = self.dxl_client.get_torque_enabled(motor_ids)
        stalled = [mid for mid, enabled in torque_state.items() if not enabled]
        if not stalled:
            return []
        error_status = self.dxl_client.get_hardware_error_status(stalled)
        print(f"Detected stalled motor(s) (torque auto-disabled): {stalled}, "
              f"hardware error status: {error_status}")
        self.dxl_client.set_torque_enabled(stalled, True, retries=2)
        still_stalled = [mid for mid, enabled in self.dxl_client.get_torque_enabled(stalled).items() if not enabled]
        if still_stalled and reboot_if_needed:
            print(f"Plain re-enable didn't clear it for {still_stalled}, rebooting...")
            self.dxl_client.reboot(still_stalled)
            time.sleep(0.5)  # let the reboot complete before touching the motor again
            self.dxl_client.sync_write(still_stalled, np.ones(len(still_stalled)) * self.kP, 84, 2)
            self.dxl_client.sync_write(still_stalled, np.ones(len(still_stalled)) * self.kI, 82, 2)
            self.dxl_client.sync_write(still_stalled, np.ones(len(still_stalled)) * self.kD, 80, 2)
            self.dxl_client.sync_write(still_stalled, np.ones(len(still_stalled)) * self.curr_lim, 102, 2)
            self.dxl_client.set_torque_enabled(still_stalled, True, retries=2)
        return stalled
#init the node
def main(mode = "realtime", **kwargs):
    # valid_motors = [8, 9, 10, 11]
    # valid_motors = [0, 1, 2, 3, 12, 13, 14, 15] # index and thumb
    valid_motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # all motors
    leap_hand = LeapNode(motors=valid_motors)
    pos = np.zeros(16)  # Initialize position array for valid motors
    if mode == "test": # run a fixed triangle wave on all motors
        step = np.deg2rad(1)
        angle_step = step  # 5 degrees in radians
        max_angle = np.deg2rad(60)  # 90 degrees in radians
        min_angle = np.deg2rad(0)  # -90 degrees in radians
        PRINT_EVERY_N_LOOPS = 33  # ~once/second at this mode's 0.03s loop period
    elif mode == "realtime": # need to run hand_joint_tracker.py to send UDP packets with joint angles
        # listen for UDP messages
        UDP_IP = "0.0.0.0"
        UDP_PORT = 5005
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        print("Listening for joint angles...")
        # Flip to -1.0 if the measured spread direction comes out mirrored on hardware
        # (e.g. a V-gesture closes the fingers instead of spreading them).
        ABDUCTION_GAIN = 1.0

        # Decouple the motor control rate from the vision/UDP rate -- the camera +
        # MediaPipe inference loop runs irregularly and can hiccup, and previously
        # recvfrom() blocked the control loop directly on it, so a slow vision frame left
        # the motors holding a stale command until the next packet, then jumping straight
        # to the new one. A background thread just keeps target_pos updated with whatever
        # arrived most recently; the control loop below runs at its own fixed pace and
        # slews toward it (see slew_limit above), so gaps/glitches on the vision side no
        # longer directly show up as jerky motor motion.
        target_pos = pos.copy()
        target_lock = threading.Lock()

        def udp_listener():
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring']
            while True:
                data, _ = sock.recvfrom(1024)
                try:
                    angles = list(map(int, data.decode().strip().split(",")))
                    new_target = target_pos.copy()
                    for i, finger in enumerate(finger_names):
                        base = i * 3
                        # Set the position for each finger based on the received angles, and add gain for certain motor angles
                        if i == 1:  # Index finger
                            new_target[1:4] = np.deg2rad([angles[base]*1.5, angles[base+1]*1, angles[base+2]*1.8])
                        elif i == 2:  # Middle finger
                            new_target[5:8] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                        elif i == 3:  # Ring finger
                            new_target[9:12] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                        elif i == 0:  # Thumb
                            # Previously copied the IP-joint (DIP-slot) angle into all three
                            # thumb flexion motors with made-up multipliers, discarding the
                            # two independently-tracked CMC/MCP angles entirely. Use each
                            # tracked joint angle for its own motor instead:
                            # angles[base]/[base+1]/[base+2] are the CMC-flex-proxy/MCP-flex/
                            # IP-flex angles (hand_joint_tracker.py's Thumb landmark order
                            # 1,2,3,4), lined up with IDs 13/14/15 (MCP_Forward/PIP/DIP per
                            # the project's documented joint layout). This is still an
                            # approximation -- the thumb's CMC joint does opposition
                            # (flexion + abduction across the palm), not a simple hinge like
                            # the other fingers' MCPs -- but it's a real per-joint signal
                            # instead of one angle faked into three.
                            new_target[13:16] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                    # MCP side (abduction/spread) for Index/Middle/Ring -- appended after the
                    # 12 flexion values as angles[12..14], since hand_joint_tracker.py's
                    # fingers dict (Thumb, Index, Middle, Ring) doesn't track it per-finger.
                    # Thumb's side joint (ID 12) isn't driven here -- its abduction is
                    # mechanically distinct (opposition, not spread) and isn't modeled yet.
                    if len(angles) >= 15:
                        new_target[0] = ABDUCTION_GAIN * np.deg2rad(angles[12])  # Index MCP side
                        new_target[4] = ABDUCTION_GAIN * np.deg2rad(angles[13])  # Middle MCP side
                        new_target[8] = ABDUCTION_GAIN * np.deg2rad(angles[14])  # Ring MCP side
                    with target_lock:
                        target_pos[:] = new_target
                except Exception as e:
                    print(f"Failed to parse data: {e}")

        threading.Thread(target=udp_listener, daemon=True).start()

        CONTROL_PERIOD_S = 0.01  # fixed 100Hz control loop, independent of UDP arrival rate
        # Caps how fast a commanded position can change per second -- generous enough not
        # to feel laggy for real gesture motion, but enough to absorb an isolated noisy or
        # garbled packet instead of jerking the motor straight to it.
        MAX_SLEW_RAD_PER_S = np.deg2rad(400)
        # The thumb's Forward/PIP/DIP motors (13/14/15 -- Side/12 isn't driven in realtime
        # mode at all, see udp_listener below) have tripped Overload (the Dynamixel
        # firmware's own protective torque cutoff, triggered by sustained high current)
        # after calibration. A lower slew cap just on these three reduces how hard the
        # position-current controller has to work to chase a fast-changing tracked target.
        # Starting point, not verified against a specific duty-cycle spec -- tune down
        # further if Overload still trips, or up if the thumb now feels too sluggish.
        THUMB_MAX_SLEW_RAD_PER_S = np.deg2rad(120)
        max_step = np.full(16, MAX_SLEW_RAD_PER_S) * CONTROL_PERIOD_S
        max_step[[13, 14, 15]] = THUMB_MAX_SLEW_RAD_PER_S * CONTROL_PERIOD_S
        PRINT_EVERY_N_LOOPS = 100  # ~once/second at this mode's 100Hz loop rate
        # How often to check for a motor the firmware auto-disabled (e.g. an Overload
        # trip from a finger clash) and try to recover it -- not every tick, since each
        # check is a handful of individual register reads across all 16 motors.
        FAULT_CHECK_EVERY_N_LOOPS = 100  # ~once/second

    loop_count = 0
    while True:
        loop_count += 1
        # Status prints are throttled so the console stays readable instead of printing
        # every single cycle.
        verbose = (loop_count % PRINT_EVERY_N_LOOPS == 0)
        try:
            # Clip to real joint limits before sending -- previously ungarded here
            # (set_allegro doesn't clip), so a miscalibrated or out-of-range tracked angle
            # (particularly a thumb joint, whose per-session calibrated motor-degree
            # window in hand_joint_tracker.py can map wider than the joint's real
            # physical range) could command the motor to strain continuously against its
            # mechanical limit under active position-current control -- itself a
            # plausible cause of an Overload trip, independent of how fast it got there.
            target_real = lhu.angle_safety_clip(lhu.allegro_to_LEAPhand(pos, zeros=False))
            leap_hand.set_leap(target_real)
            if verbose:
                print("Desired Position: " + str(pos[valid_motors]))
                print("Read Position: " + str(leap_hand.read_pos()))
            if mode == "realtime" and loop_count % FAULT_CHECK_EVERY_N_LOOPS == 0:
                leap_hand.recover_stalled_motors(valid_motors)
        except Exception as e:
            # A transient USB/serial hiccup (seen in practice as a termios.error:
            # Input/output error raised deep inside the Dynamixel SDK's txPacket)
            # previously crashed the whole script, losing all state and requiring a full
            # restart. Reconnect the port and keep going instead -- motor-side state
            # (gains, current limit, torque enable) lives on the motors themselves and
            # survives a port-level reconnect, unlike a motor reboot.
            print(f"Communication error ({e!r}) -- attempting to reconnect...")
            try:
                leap_hand.dxl_client.reconnect()
                print("Reconnected.")
            except Exception as reconnect_error:
                print(f"Reconnect failed ({reconnect_error!r}), will retry next cycle.")
            time.sleep(0.5)
            continue
        if mode == "test":
            if pos[valid_motors[0]] <= min_angle:
                angle_step = step
            if pos[valid_motors[0]] >= max_angle:
                angle_step = -1*step
            pos[valid_motors] += angle_step
            time.sleep(0.03)
        elif mode == "realtime":
            with target_lock:
                target = target_pos.copy()
            pos = slew_limit(pos, target, max_step)
            time.sleep(CONTROL_PERIOD_S)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Leap Hand Control Script")
    parser.add_argument("--mode", choices=["test", "realtime"], default="realtime",
                        help="Select operation mode: 'test' for triangle wave, 'realtime' for live UDP control")

    args = parser.parse_args()

    main(mode=args.mode)