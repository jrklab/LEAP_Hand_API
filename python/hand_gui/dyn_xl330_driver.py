import os
import time
# ### MODIFIED ### Changed to a standard import
import dynamixel_sdk

# #############################################################################
# ## Motor Constants
# #############################################################################

# Protocol version
PROTOCOL_VERSION = 2.0

# ### MODIFIED ### Default connection settings
BAUDRATE = 4000000
DEVICE_NAME = '/dev/ttyUSB0'

# ### NEW ### XL330-M288 Control Table Addresses
ADDR_TORQUE_ENABLE      = 64
ADDR_LED                = 65
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_POSITION_D_GAIN    = 80
ADDR_POSITION_I_GAIN    = 82
ADDR_POSITION_P_GAIN    = 84
ADDR_GOAL_PWM           = 100
ADDR_GOAL_CURRENT       = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

# Data Lengths
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4
LEN_PROFILE_VELOCITY    = 4
LEN_PROFILE_ACCELERATION = 4
LEN_POSITION_P_GAIN     = 2
LEN_POSITION_I_GAIN     = 2
LEN_POSITION_D_GAIN     = 2
LEN_GOAL_CURRENT        = 2
LEN_HARDWARE_ERROR_STATUS = 1

class MotorDriver:
    def __init__(self, motor_ids):
        """
        Initializes the motor driver.
        :param motor_ids: A list of motor IDs to control.
        """
        self.motor_ids = motor_ids
        # ### MODIFIED ### Added dynamixel_sdk prefix
        self.portHandler = dynamixel_sdk.PortHandler(DEVICE_NAME)
        self.packetHandler = dynamixel_sdk.PacketHandler(PROTOCOL_VERSION)

        # ### MODIFIED ### Added dynamixel_sdk prefix
        self.groupSyncWritePosition = dynamixel_sdk.GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self.groupSyncReadPosition = dynamixel_sdk.GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        # ### NEW ### GroupSyncRead instance for the error status
        self.groupSyncReadError = dynamixel_sdk.GroupSyncRead(self.portHandler, self.packetHandler, ADDR_HARDWARE_ERROR_STATUS, LEN_HARDWARE_ERROR_STATUS)

    def connect(self):
        """
        Connects to the serial port and sets the baudrate.
        """
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            return False

        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            return False
        return True

    def disconnect(self):
        """
        Closes the serial port.
        """
        self.portHandler.closePort()

    def enable_torque(self):
        """
        Enables torque for all motors.
        """
        for motor_id in self.motor_ids:
            # ### MODIFIED ### Added dynamixel_sdk prefix
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, ADDR_TORQUE_ENABLE, 1)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f"ID {motor_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"ID {motor_id}: {self.packetHandler.getRxPacketError(dxl_error)}")
            else:
                print(f"Dynamixel#{motor_id} has been successfully connected")

    def disable_torque(self):
        """
        Disables torque for all motors.
        """
        for motor_id in self.motor_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)

    def initialize_registers(self, p_gain, i_gain, d_gain, goal_current, profile_accel, profile_vel):
        """
        Initializes PID gains, goal current, and profile settings for all motors.
        """
        print("Initializing registers...")
        for motor_id in self.motor_ids:
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_P_GAIN, p_gain)
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_I_GAIN, i_gain)
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_POSITION_D_GAIN, d_gain)
            self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, ADDR_GOAL_CURRENT, goal_current)
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, ADDR_PROFILE_ACCELERATION, profile_accel)
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, ADDR_PROFILE_VELOCITY, profile_vel)
        print("Register initialization complete.")


    def set_goal_positions(self, positions):
        """
        Sets the goal positions for all motors simultaneously.
        :param positions: A dictionary mapping motor IDs to goal positions.
        """
        self.groupSyncWritePosition.clearParam()
        for motor_id, position in positions.items():
            # ### MODIFIED ### Added dynamixel_sdk prefix to utility functions
            param_goal_position = [
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(position)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(position)),
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(position)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(position))
            ]
            self.groupSyncWritePosition.addParam(motor_id, param_goal_position)
        
        # ### MODIFIED ### Added dynamixel_sdk prefix
        dxl_comm_result = self.groupSyncWritePosition.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")

    def read_present_positions(self):
        """
        Reads the present positions of all motors simultaneously.
        :return: A dictionary mapping motor IDs to their present positions.
        """
        self.groupSyncReadPosition.clearParam()
        for motor_id in self.motor_ids:
            self.groupSyncReadPosition.addParam(motor_id)

        # ### MODIFIED ### Added dynamixel_sdk prefix
        dxl_comm_result = self.groupSyncReadPosition.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return None

        positions = {}
        for motor_id in self.motor_ids:
            if self.groupSyncReadPosition.isAvailable(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                position = self.groupSyncReadPosition.getData(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                positions[motor_id] = position
        return positions
        # ### NEW ### Method to read hardware error status for all motors
    def read_hardware_error_status(self):
        """
        Reads the hardware error status of all motors simultaneously.
        :return: A dictionary mapping motor IDs to their error codes.
        """
        self.groupSyncReadError.clearParam()
        for motor_id in self.motor_ids:
            self.groupSyncReadError.addParam(motor_id)

        dxl_comm_result = self.groupSyncReadError.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print(f"{self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return None
        
        errors = {}
        for motor_id in self.motor_ids:
            if self.groupSyncReadError.isAvailable(motor_id, ADDR_HARDWARE_ERROR_STATUS, LEN_HARDWARE_ERROR_STATUS):
                error_code = self.groupSyncReadError.getData(motor_id, ADDR_HARDWARE_ERROR_STATUS, LEN_HARDWARE_ERROR_STATUS)
                errors[motor_id] = error_code
        return errors

if __name__ == '__main__':
    MOTOR_IDS = list(range(0, 16))
    driver = MotorDriver(MOTOR_IDS)

    if driver.connect():
        INITIAL_P_GAIN = 600
        INITIAL_I_GAIN = 0
        INITIAL_D_GAIN = 200
        INITIAL_GOAL_CURRENT = 350
        INITIAL_PROFILE_ACCEL = 10
        INITIAL_PROFILE_VEL = 100

        driver.initialize_registers(
            p_gain=INITIAL_P_GAIN,
            i_gain=INITIAL_I_GAIN,
            d_gain=INITIAL_D_GAIN,
            goal_current=INITIAL_GOAL_CURRENT,
            profile_accel=INITIAL_PROFILE_ACCEL,
            profile_vel=INITIAL_PROFILE_VEL
        )
        time.sleep(1)

        driver.enable_torque()

        # goal_positions = {motor_id: 1000 for motor_id in MOTOR_IDS}
        # driver.set_goal_positions(goal_positions)
        # print("Set goal positions to 1000")
        # time.sleep(2)

        # present_positions = driver.read_present_positions()
        # if present_positions:
        #     for motor_id, position in present_positions.items():
        #         print(f"Motor {motor_id} is at position {position}")

        goal_positions = {motor_id: 2048 for motor_id in MOTOR_IDS}
        print(goal_positions)
        driver.set_goal_positions(goal_positions)
        print("Set goal positions to 2000")
        time.sleep(2)
        
        present_positions = driver.read_present_positions()
        if present_positions:
            for motor_id, position in present_positions.items():
                print(f"Motor {motor_id} is at position {position}")

        driver.disable_torque()
        driver.disconnect()