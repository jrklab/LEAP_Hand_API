import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from hand_driver import HandDriver

CURL_LOW_LIMIT = 0.20

class HandControlGUI(tk.Tk):
    def __init__(self, hand_driver):
        super().__init__()
        self.title("LeapHand GUI Controller")
        self.geometry("1400x900")
        self.hand = hand_driver
        self.comm_lock = threading.Lock()
        self.is_running = True
        self.super_control_vars = {}
        self.sliders = {}
        self.error_labels = {}

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        control_tab_frame = ttk.Frame(notebook, padding=10)
        poses_tab_frame = ttk.Frame(notebook, padding=10)

        notebook.add(control_tab_frame, text='Control')
        notebook.add(poses_tab_frame, text='Poses')

        self._create_control_tab(control_tab_frame)
        self._create_poses_tab(poses_tab_frame)
        
        self.set_registers()
        self.set_torque()

        self.readout_thread = threading.Thread(target=self._update_readouts, daemon=True)
        self.readout_thread.start()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_control_tab(self, parent):
        """Populates the Control tab with register and joint controls."""
        register_frame = ttk.LabelFrame(parent, text="Motor Registers", padding=10)
        register_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        control_frame = ttk.LabelFrame(parent, text="Joint Control", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_register_controls(register_frame)
        self._create_joint_controls(control_frame)

    def _create_poses_tab(self, parent):
        poses = {"Fist": self.pose_fist, "OK": self.pose_ok, "Peace": self.pose_peace, "Pan": self.pose_pan, "Idle": self.pose_idle}
        button_frame = ttk.Frame(parent)
        button_frame.pack(anchor="w")
        for name, command in poses.items():
            button = ttk.Button(button_frame, text=name, command=command)
            button.pack(side=tk.LEFT, padx=5, pady=5)

    def _create_register_controls(self, parent):
        """Creates input boxes for motor register settings."""
        self.register_vars = {}
        registers = {"p_gain": 800, "i_gain": 0, "d_gain": 600, "goal_current": 200, "profile_accel": 25, "profile_vel": 50}
        
        for i, (name, default_val) in enumerate(registers.items()):
            ttk.Label(parent, text=name.replace('_', ' ').title() + ":").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(default_val))
            self.register_vars[name] = var
            ttk.Entry(parent, textvariable=var, width=10).grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            
        set_btn = ttk.Button(parent, text="Set Registers", command=self.set_registers)
        set_btn.grid(row=len(registers), column=0, columnspan=2, pady=10)

        self.torque_var = tk.BooleanVar(value=True)
        torque_check = ttk.Checkbutton(parent, text="Torque Enable", variable=self.torque_var, command=self.set_torque)
        torque_check.grid(row=len(registers)+1, column=0, columnspan=2, pady=10)

    def set_registers(self):
        try:
            settings = {name: int(var.get()) for name, var in self.register_vars.items()}
            with self.comm_lock:
                self.hand.set_motor_registers(**settings)
        except ValueError:
            messagebox.showerror("Error", "All register values must be integers.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            
    def set_torque(self):
        with self.comm_lock:
            self.hand.set_torque(self.torque_var.get())

    def _create_joint_controls(self, parent):
        self.readout_labels = {}
        
        super_control_frame = ttk.LabelFrame(parent, text="Super Control", padding=10)
        super_control_frame.pack(fill=tk.X, expand=True, pady=5)
        
        default_super_val = 0.5
        self.super_slider_var = tk.DoubleVar(value=default_super_val)
        self.super_slider_label = ttk.Label(super_control_frame, text=f"{default_super_val:.2f}", width=5)
        self.super_slider_label.pack(side=tk.LEFT, padx=(5,0))
        super_slider = ttk.Scale(super_control_frame, from_=0, to=1.0, orient=tk.HORIZONTAL, variable=self.super_slider_var, command=self.on_super_slider_move)
        super_slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)

        fingers = ['thumb', 'index', 'middle', 'ring']
        for finger in fingers:
            finger_frame = ttk.LabelFrame(parent, text=finger.title(), padding=10)
            finger_frame.pack(fill=tk.X, expand=True, pady=5)
            
            # ### MODIFIED ### Use grid layout for precise alignment and expansion
            finger_frame.columnconfigure(0, weight=1) # Makes the left column (curl) expand
            finger_frame.columnconfigure(1, weight=1) # Makes the right column (wiggle) expand

            # Create and place Curl controls in the left column
            curl_frame = ttk.LabelFrame(finger_frame, text="Curl", padding=5)
            curl_frame.grid(row=0, column=0, sticky="nsew", padx=5, rowspan=3)
            self._create_slider(curl_frame, f"{finger} pip", default_value=1.0)
            self._create_slider(curl_frame, f"{finger} dip", default_value=1.0)
            
            # Create and place Wiggle controls in the right column
            wiggle_frame = ttk.LabelFrame(finger_frame, text="Wiggle", padding=5)
            wiggle_frame.grid(row=0, column=1, sticky="nsew", padx=5, rowspan=3)

            # Create MCP controls in their respective frames to align them
            self._create_slider(curl_frame, f"{finger} mcp_forward", default_value=1.0)
            self._create_slider(wiggle_frame, f"{finger} mcp_side", default_value=0.5)

    def _create_slider(self, parent, joint_name, default_value):
        if joint_name not in self.hand.joint_names:
            return
            
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, expand=True, pady=2)
        
        super_var = tk.BooleanVar(value=False)
        checkbox = ttk.Checkbutton(container, variable=super_var, command=lambda j=joint_name: self.toggle_super_control(j, super_var.get()))
        checkbox.pack(side=tk.LEFT)
        self.super_control_vars[joint_name] = super_var
        
        joint_part_name = joint_name.split(' ')[-1].upper()
        if joint_part_name == 'MCP_FORWARD':
            joint_part_name = 'MCP_FWD'
        ttk.Label(container, text=joint_part_name, width=8).pack(side=tk.LEFT)
        
        set_var = tk.DoubleVar(value=default_value)
        set_label = ttk.Label(container, text=f"{default_value:.2f}", width=5)
        set_label.pack(side=tk.LEFT, padx=(5,0))
        
        slider = ttk.Scale(container, from_=0, to=1.0, orient=tk.HORIZONTAL, variable=set_var, command=lambda val, j=joint_name, sl=set_label: self.on_slider_move(j, float(val), sl))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sliders[joint_name] = slider
        
        readout_label = ttk.Label(container, text="--", width=5)
        readout_label.pack(side=tk.LEFT, padx=(0,5))
        self.readout_labels[joint_name] = readout_label
        
        error_label = ttk.Label(container, text="OK", width=15, foreground="green")
        error_label.pack(side=tk.LEFT, padx=(5,0))
        self.error_labels[joint_name] = error_label

        self.set_joint_position(joint_name, default_value)
        
    def toggle_super_control(self, joint_name, is_super_controlled):
        slider = self.sliders.get(joint_name)
        if slider:
            slider.config(state=tk.DISABLED if is_super_controlled else tk.NORMAL)
            if is_super_controlled:
                self.on_super_slider_move(self.super_slider_var.get())

    def on_super_slider_move(self, value):
        float_value = float(value)
        self.super_slider_label.config(text=f"{float_value:.2f}")
        positions_to_set = {}
        for joint_name, var in self.super_control_vars.items():
            if var.get():
                positions_to_set[joint_name] = float_value
        if positions_to_set:
            self.set_multiple_joint_positions(positions_to_set)

    def on_slider_move(self, joint_name, value, set_label):
        set_label.config(text=f"{value:.2f}")
        self.set_joint_position(joint_name, value)

    def set_joint_position(self, joint_name, value):
        if not self.torque_var.get(): return
        def task():
            with self.comm_lock:
                self.hand.set_hand_position({joint_name: value})
        threading.Thread(target=task, daemon=True).start()

    def set_multiple_joint_positions(self, positions):
        if not self.torque_var.get(): return
        def task():
            with self.comm_lock:
                self.hand.set_hand_position(positions)
        threading.Thread(target=task, daemon=True).start()

    def _update_readouts(self):
        while self.is_running:
            with self.comm_lock:
                positions = self.hand.read_hand_position()
                errors = self.hand.read_hardware_error_status()
            
            if positions: self.after(0, self._update_labels, positions)
            if errors: self.after(0, self._update_error_labels, errors)
            time.sleep(1.0 / 10)

    def _update_labels(self, positions):
        for joint_name, label in self.readout_labels.items():
            if joint_name in positions:
                label.config(text=f"{positions[joint_name]:.2f}")

    def _update_error_labels(self, errors):
        for joint_name, label in self.error_labels.items():
            if joint_name in errors:
                error_code = errors[joint_name]
                error_text = self._parse_error_code(error_code)
                color = "red" if error_text != "OK" else "green"
                label.config(text=error_text, foreground=color)

    def _parse_error_code(self, code):
        if code is None or code == 0:
            return "OK"
        
        error_list = []
        if code & 1: error_list.append("Input Voltage")
        if code & 4: error_list.append("Overheating")
        if code & 16: error_list.append("Electrical Shock")
        if code & 32: error_list.append("Overload")
        
        return ", ".join(error_list) if error_list else "OK"
    
    # def _execute_pose(self, pose_func):
    #     threading.Thread(target=pose_func, daemon=True).start()
    # def pose_fist(self): self._execute_pose(lambda: self.comm_lock.acquire() and (self.hand.index_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.middle_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.thumb_curl(0.2, 0.2, 0.2), self.hand.thumb_wiggle(0), self.comm_lock.release()))
    # def pose_ok(self): self._execute_pose(lambda: self.comm_lock.acquire() and (self.hand.index_curl(0.25, 0.25, 0.25), self.hand.middle_curl(1, 1, 1), self.hand.ring_curl(1, 1, 1), self.hand.thumb_curl(0.28, 0.28, 0.28), self.hand.thumb_wiggle(0.6), self.comm_lock.release()))
    # def pose_peace(self): self._execute_pose(lambda: self.comm_lock.acquire() and (self.hand.index_curl(1, 1, 1), self.hand.middle_curl(1, 1, 1), self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.thumb_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.comm_lock.release()))
    # def pose_pan(self): self._execute_pose(lambda: self.comm_lock.acquire() and (self.hand.index_curl(1, 1, 1), self.hand.middle_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.hand.thumb_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT), self.comm_lock.release()))
    # def pose_idle(self): self._execute_pose(lambda: self.comm_lock.acquire() and (self.hand.index_curl(1, 1, 1), self.hand.middle_curl(1, 1, 1), self.hand.ring_curl(1, 1, 1), self.hand.thumb_curl(1, 1, 1), self.hand.thumb_wiggle(0.5), self.comm_lock.release()))
    # def on_closing(self):
    #     print("Closing application..."); self.is_running = False; self.readout_thread.join(timeout=0.5); self.hand.close(); self.destroy()

    def _execute_pose(self, pose_func):
        """Wrapper to run a pose function in a thread."""
        threading.Thread(target=pose_func, daemon=True).start()

    def pose_fist(self):
        def task():
            with self.comm_lock:
                self.hand.index_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                self.hand.middle_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                time.sleep(0.2)
                self.hand.thumb_curl(0.4, 0.4, 0.4)
                self.hand.thumb_wiggle(0.6)
        self._execute_pose(task)

    def pose_ok(self):
        def task():
            with self.comm_lock:
                self.hand.index_curl(0.52, 0.52, 0.52)
                self.hand.middle_curl(1, 1, 1)
                self.hand.ring_curl(1, 1, 1)
                self.hand.thumb_curl(0.54, 0.54, 0.54)
                self.hand.thumb_wiggle(0.42)
        self._execute_pose(task)

    def pose_peace(self):
        def task():
            with self.comm_lock:
                self.hand.index_curl(1, 1, 1)
                self.hand.middle_curl(1, 1, 1)
                self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                self.hand.thumb_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
        self._execute_pose(task)

    def pose_pan(self):
        def task():
            with self.comm_lock:
                self.hand.index_curl(1, 1, 1)
                self.hand.middle_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                self.hand.ring_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
                self.hand.thumb_curl(CURL_LOW_LIMIT, CURL_LOW_LIMIT, CURL_LOW_LIMIT)
        self._execute_pose(task)

    def pose_idle(self):
        def task():
            with self.comm_lock:
                self.hand.index_curl(1, 1, 1)
                self.hand.middle_curl(1, 1, 1)
                self.hand.ring_curl(1, 1, 1)
                self.hand.thumb_curl(1, 1, 1)
                self.hand.thumb_wiggle(0.5)
        self._execute_pose(task)

    def on_closing(self):
        """Handles window close event to safely shut down."""
        print("Closing application...")
        self.is_running = False
        self.readout_thread.join(timeout=0.5)
        self.hand.close()
        self.destroy()

if __name__ == '__main__':
    try:
        hand = HandDriver()
        app = HandControlGUI(hand)
        app.mainloop()
    except (ValueError, ConnectionError) as e:
        messagebox.showerror("Initialization Failed", f"Could not start the application: {e}")
    except Exception as e:
        messagebox.showerror("An Unexpected Error Occurred", f"An error occurred: {e}")