import os
import time
import dynamixel_sdk

# #############################################################################
# ## Motor Constants
# #############################################################################

PROTOCOL_VERSION = 2.0
BAUDRATE = 4000000

# Control Table Addresses for XL330-M288-T
ADDR_TORQUE_ENABLE      = 64
ADDR_POSITION_D_GAIN    = 80
ADDR_POSITION_I_GAIN    = 82
ADDR_POSITION_P_GAIN    = 84
ADDR_GOAL_CURRENT       = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4

class MotorMimicController:
    """Manages two sets of motors (leader and follower) on a single serial port."""
    def __init__(self, device_name, leader_ids, follower_ids):
        self.portHandler = dynamixel_sdk.PortHandler(device_name)
        self.packetHandler = dynamixel_sdk.PacketHandler(PROTOCOL_VERSION)
        
        self.leader_ids = leader_ids
        self.follower_ids = follower_ids
        self.id_map = dict(zip(self.leader_ids, self.follower_ids))

        # A reader for the leader set's position
        self.groupSyncReadLeader = dynamixel_sdk.GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        # A writer for the follower set's position
        self.groupSyncWriteFollower = dynamixel_sdk.GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def connect(self):
        """Connects to the serial port."""
        if not self.portHandler.openPort():
            print(f"Failed to open port {self.portHandler.port_name}")
            return False
        if not self.portHandler.setBaudRate(BAUDRATE):
            print(f"Failed to set baudrate to {BAUDRATE}")
            return False
        print(f"Successfully connected to port {self.portHandler.port_name}")
        return True

    def disconnect(self):
        """Closes the serial port."""
        self.portHandler.closePort()

    def initialize_registers(self, leader_settings, follower_settings):
        """Initializes registers for both sets of motors."""
        print("Initializing leader set registers...")
        for motor_id in self.leader_ids:
            self._set_motor_regs(motor_id, leader_settings)
            
        print("Initializing follower set registers...")
        for motor_id in self.follower_ids:
            self._set_motor_regs(motor_id, follower_settings)
        print("Register initialization complete.")

    def _set_motor_regs(self, motor_id, settings):
        """Helper to write register values to a single motor."""
        self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_P_GAIN, settings['p_gain'])
        self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_I_GAIN, settings['i_gain'])
        self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_D_GAIN, settings['d_gain'])
        self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_GOAL_CURRENT, settings['goal_current'])
        self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, ADDR_PROFILE_ACCELERATION, settings['profile_accel'])
        self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, ADDR_PROFILE_VELOCITY, settings['profile_vel'])

    def set_torque(self, motor_ids, enable):
        """Enables or disables torque for a specific list of motor IDs."""
        status_text = "Enabling" if enable else "Disabling"
        print(f"{status_text} torque for motor IDs: {motor_ids}...")
        for motor_id in motor_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def run_mimic_loop(self, frequency_hz=20):
        """Starts the main loop to read from leaders and write to followers."""
        print("\n*** Starting motor mimic loop. Press Ctrl+C to exit. ***")
        while True:
            # 1. Read positions from the leader set
            leader_positions = self._read_leader_positions()

            if leader_positions:
                # 2. Create the goal position dictionary for the follower set
                follower_goals = {self.id_map[leader_id]: pos for leader_id, pos in leader_positions.items()}
                
                # 3. Write the positions to the follower set
                self._write_follower_positions(follower_goals)
            
            # 4. Wait to maintain the desired frequency
            time.sleep(1.0 / frequency_hz)

    def _read_leader_positions(self):
        self.groupSyncReadLeader.clearParam()
        for motor_id in self.leader_ids:
            self.groupSyncReadLeader.addParam(motor_id)
        
        dxl_comm_result = self.groupSyncReadLeader.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"Leader SyncRead failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return None

        positions = {}
        for motor_id in self.leader_ids:
            if self.groupSyncReadLeader.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                positions[motor_id] = self.groupSyncReadLeader.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        
        # Return only if all motors in the set responded
        return positions if len(positions) == len(self.leader_ids) else None

    def _write_follower_positions(self, positions):
        self.groupSyncWriteFollower.clearParam()
        for motor_id, position in positions.items():
            param_goal = [
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(position)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(position)),
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(position)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(position))
            ]
            self.groupSyncWriteFollower.addParam(motor_id, param_goal)
        
        dxl_comm_result = self.groupSyncWriteFollower.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"Follower SyncWrite failed: {self.packetHandler.getTxRxResult(dxl_comm_result)}")

if __name__ == '__main__':
    # #########################################################################
    # ## USER CONFIGURATION
    # #########################################################################
    
    # --- Serial Port ---
    DEVICE_PORT = '/dev/ttyUSB0'
    
    # --- Motor IDs (MUST BE UNIQUE AND NOT OVERLAP) ---
    LEADER_IDS = list([0, 1, 2, 3])   # index finger, used as the leader
    FOLLOWER_IDS = list([8, 9, 10, 11]) # ring finger, used as the follower
    MIMIC_FREQUENCY_HZ = 40

    LEADER_REGISTER_SETTINGS = {
        "p_gain": 800, "i_gain": 0, "d_gain": 600,
        "goal_current": 10, "profile_accel": 25, "profile_vel": 200
    } # small goal_current to provide a little bit of torque to maintain a stable position
    FOLLOWER_REGISTER_SETTINGS = {
        "p_gain": 800, "i_gain": 0, "d_gain": 600,
        "goal_current": 200, "profile_accel": 100, "profile_vel": 200
    }
    
    # #########################################################################
    
    controller = MotorMimicController(DEVICE_PORT, LEADER_IDS, FOLLOWER_IDS)

    if not controller.connect():
        exit()

    controller.initialize_registers(LEADER_REGISTER_SETTINGS, FOLLOWER_REGISTER_SETTINGS)
    
    # --- IMPORTANT ---
    controller.set_torque(controller.leader_ids, enable=True) # Disable torque on leaders
    controller.set_torque(controller.follower_ids, enable=True)  # Enable torque on followers
    
    try:
        controller.run_mimic_loop(MIMIC_FREQUENCY_HZ)
    except KeyboardInterrupt:
        print("\n*** Exiting program. ***")
    finally:
        print("Disabling torque on all motors...")
        controller.set_torque(controller.follower_ids, enable=False)
        controller.disconnect()