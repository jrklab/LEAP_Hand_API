import os
import time
# ### MODIFIED ### Changed to a standard import
import dynamixel_sdk
import numpy as np

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
# Firmware auto-disables ADDR_TORQUE_ENABLE and latches a bit here when it detects a
# fault (bit 5 = Overload -- sustained high current, e.g. from a mechanical clash
# against another finger; bit 4 = Electrical Shock; bit 3 = Encoder; bit 2 =
# Overheating; bit 0 = Input Voltage). See DynamixelClient.get_hardware_error_status /
# recover_stalled_motors below.
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_POSITION_D_GAIN    = 80
ADDR_POSITION_I_GAIN    = 82
ADDR_POSITION_P_GAIN    = 84
ADDR_GOAL_PWM           = 100
ADDR_GOAL_CURRENT       = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY   = 112
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_CURRENT    = 126
ADDR_PRESENT_VELOCITY   = 128
ADDR_PRESENT_POSITION   = 132

# Data Lengths
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4
LEN_PRESENT_VELOCITY    = 4
LEN_PRESENT_CURRENT     = 2
LEN_PROFILE_VELOCITY    = 4
LEN_PROFILE_ACCELERATION = 4
LEN_POSITION_P_GAIN     = 2
LEN_POSITION_I_GAIN     = 2
LEN_POSITION_D_GAIN     = 2
LEN_GOAL_CURRENT        = 2

# Ticks per revolution (XC330/XL330 and other X-series motors used here). pi radians (=half
# a revolution) is this project's documented "fully open/straight" reference pose for every
# joint (see leap_hand_utils.py's module docstring and allegro_to_LEAPhand()) -- on a motor
# whose Homing_Offset has been correctly calibrated (see ../calibrate_motor_offsets.py), that
# pose reads as exactly RESOLUTION/2 = 2048 raw ticks.
RESOLUTION = 4096


def _ticks_to_radians(ticks):
    return np.asarray(ticks, dtype=np.float64) * (2 * np.pi / RESOLUTION)


def _radians_to_ticks(radians):
    return np.round(np.asarray(radians, dtype=np.float64) * (RESOLUTION / (2 * np.pi))).astype(np.int64)


class DynamixelClient:
    """Restores the interface hand_demo.py and main.py expect: DynamixelClient(motor_ids,
    port, baudrate), .connect(), .sync_write(motor_ids, values, address, size),
    .set_torque_enabled(motor_ids, enable, retries), .write_desired_pos(motor_ids, radians),
    and .read_pos()/.read_vel()/.read_cur()/.read_pos_vel()/.read_pos_vel_cur().

    A prior commit ("modify dynamixel driver") replaced this class with the simpler
    MotorDriver below, without updating the scripts that still import DynamixelClient by
    name -- they've been broken (NameError at the moment they try to connect) ever since.
    MotorDriver is left as-is (unused elsewhere in this repo, but harmless).

    Works in radians, matching leap_hand_utils.py's convention (pi radians = raw tick 2048 =
    the documented "fully open/straight" pose), converting to/from each motor's raw ticks
    internally -- see RESOLUTION/_ticks_to_radians/_radians_to_ticks above. Deliberately has
    no per-motor offset/correction logic of its own: a motor whose horn was mounted off by a
    90-degree increment is corrected directly in its own Homing_Offset register instead (see
    calibrate_motor_offsets.py), which the motor firmware applies transparently to every
    read/write -- so this class can stay generic and doesn't need to know which motors (if
    any) needed that correction.
    """

    def __init__(self, motor_ids, port=DEVICE_NAME, baudrate=BAUDRATE):
        self.motor_ids = list(motor_ids)
        self.port_handler = dynamixel_sdk.PortHandler(port)
        self.packet_handler = dynamixel_sdk.PacketHandler(PROTOCOL_VERSION)
        self.baudrate = baudrate
        # Persistent group objects for the two operations the hot control loop calls every
        # tick (write_desired_pos/read_pos), to avoid reconstructing them every iteration --
        # same pattern MotorDriver below already uses for its own read/write groups.
        self._write_pos_group = dynamixel_sdk.GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self._read_pos_group = dynamixel_sdk.GroupSyncRead(
            self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )

    def connect(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port_handler.port}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.baudrate}")
        return True

    def disconnect(self):
        self.port_handler.closePort()

    @staticmethod
    def _int_to_bytes(value, size):
        value = int(value) & (0xFFFFFFFF if size == 4 else 0xFFFF if size == 2 else 0xFF)
        if size == 4:
            return [
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(value)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(value)),
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(value)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(value)),
            ]
        elif size == 2:
            return [dynamixel_sdk.DXL_LOBYTE(value), dynamixel_sdk.DXL_HIBYTE(value)]
        return [value]

    def sync_write(self, motor_ids, values, address, size):
        """Generic register write -- `values` is a per-motor array/list (same length and
        order as motor_ids) already in raw register units, not radians. Matches
        hand_demo.py's existing usage for Operating_Mode/PID gains/current limit at startup
        -- infrequent enough that a fresh GroupSyncWrite per call (rather than a persistent
        one) is simpler and not worth optimizing."""
        group = dynamixel_sdk.GroupSyncWrite(self.port_handler, self.packet_handler, address, size)
        for motor_id, value in zip(motor_ids, values):
            group.addParam(motor_id, self._int_to_bytes(value, size))
        group.txPacket()
        group.clearParam()

    def set_torque_enabled(self, motor_ids, enable, retries=0):
        value = 1 if enable else 0
        for motor_id in motor_ids:
            for attempt in range(retries + 1):
                result, _error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, value
                )
                if result == dynamixel_sdk.COMM_SUCCESS:
                    break

    def get_torque_enabled(self, motor_ids):
        """Returns {motor_id: bool}, read fresh from each motor rather than tracked in
        software -- the firmware can flip this to False on its own (see
        ADDR_HARDWARE_ERROR_STATUS above), independent of anything this class commanded.
        A motor whose read fails entirely (e.g. a bus glitch) is omitted rather than
        guessed at."""
        result = {}
        for motor_id in motor_ids:
            val, comm_result, _error = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE
            )
            if comm_result == dynamixel_sdk.COMM_SUCCESS:
                result[motor_id] = bool(val)
        return result

    def get_hardware_error_status(self, motor_ids):
        """Returns {motor_id: raw_byte} of ADDR_HARDWARE_ERROR_STATUS -- for diagnosing
        *why* a motor's torque was auto-disabled, not just that it was."""
        result = {}
        for motor_id in motor_ids:
            val, comm_result, _error = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, ADDR_HARDWARE_ERROR_STATUS
            )
            if comm_result == dynamixel_sdk.COMM_SUCCESS:
                result[motor_id] = val
        return result

    def reboot(self, motor_ids):
        """Reboots the given motors -- clears a latched hardware error that a plain
        Torque_Enable re-write sometimes won't clear on its own. This resets that motor's
        RAM-area settings (PID gains, current limit, goal position, torque enable) to
        firmware defaults; callers must re-apply those (and re-enable torque) afterward."""
        for motor_id in motor_ids:
            self.packet_handler.reboot(self.port_handler, motor_id)

    def write_desired_pos(self, motor_ids, radian_positions):
        ticks = _radians_to_ticks(radian_positions)
        self._write_pos_group.clearParam()
        for motor_id, tick in zip(motor_ids, ticks):
            self._write_pos_group.addParam(motor_id, self._int_to_bytes(tick, LEN_GOAL_POSITION))
        self._write_pos_group.txPacket()

    def _sync_read_raw(self, motor_ids, address, size, group=None):
        owns_group = group is None
        if owns_group:
            group = dynamixel_sdk.GroupSyncRead(self.port_handler, self.packet_handler, address, size)
            for motor_id in motor_ids:
                group.addParam(motor_id)
        group.txRxPacket()
        values = []
        for motor_id in motor_ids:
            raw = group.getData(motor_id, address, size)
            if size == 4 and raw > 0x7FFFFFFF:
                raw -= 0x100000000
            elif size == 2 and raw > 0x7FFF:
                raw -= 0x10000
            values.append(raw)
        if owns_group:
            group.clearParam()
        return np.array(values, dtype=np.float64)

    def read_pos(self):
        """Returns radians, in self.motor_ids order."""
        self._read_pos_group.clearParam()
        for motor_id in self.motor_ids:
            self._read_pos_group.addParam(motor_id)
        ticks = self._sync_read_raw(self.motor_ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION, group=self._read_pos_group)
        return _ticks_to_radians(ticks)

    def read_vel(self):
        """Returns rad/s. X-series Present_Velocity unit is 0.229 rev/min per count."""
        raw = self._sync_read_raw(self.motor_ids, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
        return raw * 0.229 * (2 * np.pi / 60.0)

    def read_cur(self):
        """Returns raw Present_Current units (not used by hand_demo.py's own control flow --
        provided for completeness/other callers). ~2.69 mA per unit on most X-series
        current-capable models, but this varies by model; left as raw units rather than
        guessing a specific model's scale factor."""
        return self._sync_read_raw(self.motor_ids, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)

    def read_pos_vel(self):
        return self.read_pos(), self.read_vel()

    def read_pos_vel_cur(self):
        return self.read_pos(), self.read_vel(), self.read_cur()


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

if __name__ == '__main__':
    MOTOR_IDS = list(range(1, 17))
    driver = MotorDriver(MOTOR_IDS)

    if driver.connect():
        INITIAL_P_GAIN = 800
        INITIAL_I_GAIN = 0
        INITIAL_D_GAIN = 0
        INITIAL_GOAL_CURRENT = 150
        INITIAL_PROFILE_ACCEL = 10
        INITIAL_PROFILE_VEL = 200

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

        goal_positions = {motor_id: 1000 for motor_id in MOTOR_IDS}
        driver.set_goal_positions(goal_positions)
        print("Set goal positions to 1000")
        time.sleep(2)

        present_positions = driver.read_present_positions()
        if present_positions:
            for motor_id, position in present_positions.items():
                print(f"Motor {motor_id} is at position {position}")

        goal_positions = {motor_id: 2000 for motor_id in MOTOR_IDS}
        driver.set_goal_positions(goal_positions)
        print("Set goal positions to 2000")
        time.sleep(2)
        
        present_positions = driver.read_present_positions()
        if present_positions:
            for motor_id, position in present_positions.items():
                print(f"Motor {motor_id} is at position {position}")

        driver.disable_torque()
        driver.disconnect()