import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
import socket
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
    while True:
        #Set to an open pose and read the joint angles 33hz
        leap_hand.set_allegro(pos)
        print("Desired Position: " + str(pos[valid_motors]))
        print("Read Position: " + str(leap_hand.read_pos()))
        if mode == "test":
            if pos[valid_motors[0]] <= min_angle:
                angle_step = step
            if pos[valid_motors[0]] >= max_angle:
                angle_step = -1*step
            pos[valid_motors] += angle_step
            time.sleep(0.03)
        elif mode == "realtime":
            data, _ = sock.recvfrom(1024)
            try:
                angles = list(map(int, data.decode().strip().split(",")))
                print("Received angles:")
                fingers = ['Thumb', 'Index', 'Middle', 'Ring']
                for i, finger in enumerate(fingers):
                    base = i * 3
                    print(f"  {finger}: MCP={angles[base]} PIP={angles[base+1]} DIP={angles[base+2]}")
                    # Set the position for each finger based on the received angles, and add gain for certain motor angles
                    if i == 1:  # Index finger
                        pos[1:4] = np.deg2rad([angles[base]*1.5, angles[base+1]*1, angles[base+2]*1.8])
                    elif i == 2:  # Middle finger
                        pos[5:8] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                    elif i == 3:  # Ring finger
                        pos[9:12] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                    elif i == 0: # Thumb
                        # Previously copied the IP-joint (DIP-slot) angle into all three
                        # thumb flexion motors with made-up multipliers, discarding the two
                        # independently-tracked CMC/MCP angles entirely. Use each tracked
                        # joint angle for its own motor instead: angles[base]/[base+1]/[base+2]
                        # are the CMC-flex-proxy/MCP-flex/IP-flex angles (hand_joint_tracker.py's
                        # Thumb landmark order 1,2,3,4), lined up with IDs 13/14/15
                        # (MCP_Forward/PIP/DIP per the project's documented joint layout).
                        # This is still an approximation -- the thumb's CMC joint does
                        # opposition (flexion + abduction across the palm), not a simple
                        # hinge like the other fingers' MCPs -- but it's a real per-joint
                        # signal instead of one angle faked into three.
                        pos[13:16] = np.deg2rad([angles[base], angles[base+1], angles[base+2]])
                # MCP side (abduction/spread) for Index/Middle/Ring -- appended after the
                # 12 flexion values as angles[12..14], since hand_joint_tracker.py's
                # fingers dict (Thumb, Index, Middle, Ring) doesn't track it per-finger.
                # Thumb's side joint (ID 12) isn't driven here -- its abduction is
                # mechanically distinct (opposition, not spread) and isn't modeled yet.
                if len(angles) >= 15:
                    pos[0] = ABDUCTION_GAIN * np.deg2rad(angles[12])  # Index MCP side
                    pos[4] = ABDUCTION_GAIN * np.deg2rad(angles[13])  # Middle MCP side
                    pos[8] = ABDUCTION_GAIN * np.deg2rad(angles[14])  # Ring MCP side
            except Exception as e:
                print(f"Failed to parse data: {e}")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Leap Hand Control Script")
    parser.add_argument("--mode", choices=["test", "realtime"], default="realtime",
                        help="Select operation mode: 'test' for triangle wave, 'realtime' for live UDP control")

    args = parser.parse_args()

    main(mode=args.mode)