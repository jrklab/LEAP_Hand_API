import configparser
import time
from dyn_xl330_driver import MotorDriver

class HandDriver:
    def __init__(self, config_file='./python/leaphand.config'):
        """
        Initializes the hand driver by reading the config file and setting up the motor controller.
        :param config_file: Path to the leaphand configuration file.
        """
        self.config = self._read_config(config_file)
        if not self.config:
            raise ValueError("Failed to load or parse the configuration file.")
            
        self.joint_names = list(self.config.keys())
        motor_ids = [joint['id'] for joint in self.config.values()]

        # Initialize the underlying motor driver
        self.motor_driver = MotorDriver(motor_ids)
        if not self.motor_driver.connect():
            raise ConnectionError("Failed to connect to the motor controller.")
        
        # Set default initial register values
        self.set_motor_registers(
            p_gain=800, i_gain=0, d_gain=600,
            goal_current=550, profile_accel=25, profile_vel=50
        )
        self.motor_driver.enable_torque()
        print(f"Successfully loaded {len(self.joint_names)} joints.")
        print(f"Motor IDs found: {sorted(motor_ids)}")

    def _read_config(self, config_file):
        """
        Manually parses the leaphand.config file to handle spaces in section headers.
        """
        joints = {}
        current_joint_name = None
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        header = line[1:-1].strip()
                        current_joint_name = header
                        joints[current_joint_name] = {}
                    elif '=' in line and current_joint_name:
                        key, value = line.split('=', 1)
                        joints[current_joint_name][key.strip()] = int(value.strip())
            return joints
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_file}")
            return None

    def _convert_to_motor_pos(self, joint_name, normalized_pos):
        """
        Converts a normalized position (0 to 1) to a raw motor position.
        """
        normalized_pos = max(0.0, min(1.0, normalized_pos))
        
        joint_info = self.config[joint_name]
        pos_in = joint_info['in']
        pos_out = joint_info['out']
        
        return int(pos_in + (pos_out - pos_in) * normalized_pos)

    def _convert_from_motor_pos(self, joint_name, motor_pos):
        """
        Converts a raw motor position to a normalized position (0 to 1).
        """
        joint_info = self.config[joint_name]
        pos_in = joint_info['in']
        pos_out = joint_info['out']
        
        if pos_out == pos_in:
            return 0.0
            
        normalized_pos = (motor_pos - pos_in) / (pos_out - pos_in)
        return max(0.0, min(1.0, normalized_pos))

    def set_hand_position(self, positions_normalized):
        """
        Sets the position for multiple joints using normalized values.
        """
        goal_positions_raw = {}
        for joint_name, norm_pos in positions_normalized.items():
            if joint_name in self.config:
                motor_id = self.config[joint_name]['id']
                raw_pos = self._convert_to_motor_pos(joint_name, norm_pos)
                goal_positions_raw[motor_id] = raw_pos
            else:
                print(f"Warning: Joint '{joint_name}' not found in configuration.")
        
        if goal_positions_raw:
            self.motor_driver.set_goal_positions(goal_positions_raw)

    def read_hand_position(self):
        """
        Reads the current position of all joints and returns them as normalized values.
        """
        raw_positions = self.motor_driver.read_present_positions()
        if not raw_positions:
            return None
            
        normalized_positions = {}
        for joint_name, joint_info in self.config.items():
            motor_id = joint_info['id']
            if motor_id in raw_positions:
                raw_pos = raw_positions[motor_id]
                norm_pos = self._convert_from_motor_pos(joint_name, raw_pos)
                normalized_positions[joint_name] = norm_pos
        
        return normalized_positions
    # ### NEW ### Method to read and map hardware error status
    def read_hardware_error_status(self):
        """
        Reads the hardware error status of all motors and maps them to joint names.
        :return: A dictionary mapping joint names to their error codes.
        """
        raw_errors = self.motor_driver.read_hardware_error_status()
        if not raw_errors:
            return None
        
        joint_errors = {}
        for joint_name, joint_info in self.config.items():
            motor_id = joint_info['id']
            if motor_id in raw_errors:
                joint_errors[joint_name] = raw_errors[motor_id]
        return joint_errors
    def close(self):
        """
        Disables torque and disconnects from the motors.
        """
        print("Closing hand driver...")
        self.motor_driver.disable_torque()
        self.motor_driver.disconnect()
        print("Connection closed.")
    
    # ### NEW ### Added the missing set_torque method
    def set_torque(self, enable):
        """
        Enables or disables torque for all motors.
        :param enable: Boolean, True to enable torque, False to disable.
        """
        if enable:
            self.motor_driver.enable_torque()
        else:
            self.motor_driver.disable_torque()
    # ### NEW ### Function to change registers after initialization
    def set_motor_registers(self, p_gain, i_gain, d_gain, goal_current, profile_accel, profile_vel):
        """
        Sets the PID gains, goal current, and profile settings for all motors.
        Can be called at any time after initialization.
        """
        self.motor_driver.initialize_registers(
            p_gain=p_gain,
            i_gain=i_gain,
            d_gain=d_gain,
            goal_current=goal_current,
            profile_accel=profile_accel,
            profile_vel=profile_vel
        )

    def thumb_curl(self, mcp_pos, pip_pos, dip_pos):
        """Controls the thumb's mcp_forward, pip, and dip joints independently."""
        self.set_hand_position({
            'thumb mcp_forward': mcp_pos,
            'thumb pip': pip_pos,
            'thumb dip': dip_pos
        })

    def thumb_wiggle(self, pos):
        """Controls the thumb's mcp_side joint."""
        self.set_hand_position({'thumb mcp_side': pos})

    def index_curl(self, mcp_pos, pip_pos, dip_pos):
        """Controls the index finger's mcp_forward, pip, and dip joints independently."""
        self.set_hand_position({
            'index mcp_forward': mcp_pos,
            'index pip': pip_pos,
            'index dip': dip_pos
        })

    def index_wiggle(self, pos):
        """Controls the index finger's mcp_side joint."""
        self.set_hand_position({'index mcp_side': pos})

    def middle_curl(self, mcp_pos, pip_pos, dip_pos):
        """Controls the middle finger's mcp_forward, pip, and dip joints independently."""
        self.set_hand_position({
            'middle mcp_forward': mcp_pos,
            'middle pip': pip_pos,
            'middle dip': dip_pos
        })

    def middle_wiggle(self, pos):
        """Controls the middle finger's mcp_side joint."""
        self.set_hand_position({'middle mcp_side': pos})

    def ring_curl(self, mcp_pos, pip_pos, dip_pos):
        """Controls the ring finger's mcp_forward, pip, and dip joints independently."""
        self.set_hand_position({
            'ring mcp_forward': mcp_pos,
            'ring pip': pip_pos,
            'ring dip': dip_pos
        })

    def ring_wiggle(self, pos):
        """Controls the ring finger's mcp_side joint."""
        self.set_hand_position({'ring mcp_side': pos})


if __name__ == '__main__':
    import atexit
    
    try:
        # Initialize the hand driver
        hand = HandDriver()
        # Register the close function to be called on script exit
        atexit.register(hand.close)

        # ### MODIFIED ### Updated Example Usage Block
        print("\n--- Running a simple demo ---")
        
        print("Fully extending index finger with default velocity...")
        hand.index_curl(mcp_pos=1.0, pip_pos=1.0, dip_pos=1.0)
        time.sleep(2)

        print("Reading finger positions after extending...")
        current_pos = hand.read_hand_position()
        if current_pos:
            print(f"  - Index MCP Forward: {current_pos.get('index mcp_forward', 'N/A'):.2f}")
        
        print("\nPartially curling index finger...")
        hand.index_curl(mcp_pos=0.3, pip_pos=0.3, dip_pos=0.3)
        time.sleep(2)

        print("Reading finger positions after curling...")
        current_pos = hand.read_hand_position()
        if current_pos:
            print(f"  - Index MCP Forward: {current_pos.get('index mcp_forward', 'N/A'):.2f}")
        
        # ### NEW ### Demonstrate changing registers at runtime
        print("\nChanging motor settings (slower profile velocity)...")
        hand.set_motor_registers(
            p_gain=800, i_gain=0, d_gain=0,
            goal_current=150, profile_accel=10, profile_vel=50 # Slower velocity
        )
        
        print("Fully extending index finger again (should be slower)...")
        hand.index_curl(mcp_pos=1.0, pip_pos=1.0, dip_pos=1.0)
        time.sleep(3) # Wait a bit longer to see the slower movement

        print("\nDemo complete.")
        
    except (ValueError, ConnectionError) as e:
        print(f"An error occurred: {e}")
    finally:
        # The atexit module will handle closing the connection
        print("Exiting script.")