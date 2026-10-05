#!/usr/bin/env python3
"""
Calibration tool for LEAP Hand motors whose servo horn was mounted a 90/180/270-degree
spline increment off from the correct orientation -- a common assembly mistake, since the
horn's spline doesn't visually indicate which tooth is "zero".

Rather than physically remounting the horn, this writes a corrective value directly into
each motor's own Homing_Offset register (Dynamixel X-series control table address 20, 4
bytes, signed). The motor's own firmware applies that correction transparently to every
future Present_Position read and Goal_Position write, so no client code (hand_demo.py, etc.)
needs any special-casing for these specific motors -- a plain DynamixelClient talking to a
"calibrated" motor just sees correct readings.

Convention this assumes (see leap_hand_utils.py's module docstring): raw position 2048
ticks (= pi radians, out of a 4096-tick/revolution encoder) is the hand's documented
"fully open/straight" reference pose for every joint.

How it works:
  1. For each motor, reads the CURRENT raw sensor value (backing out whatever Homing_Offset
     is already set, so this is safe to re-run / re-calibrate at any time -- it always
     starts from the true physical sensor reading, never compounds on a previous guess).
  2. Computes how far that is from the expected 2048, in ticks.
  3. Rounds that distance to the nearest 90-degree (1024-tick) multiple -- since a mount
     error only ever happens in 90-degree spline increments; any smaller residual is just
     imprecision in how precisely "straight" the finger was actually held during
     calibration, not a real mechanical offset, and should NOT be compensated away.
  4. Writes the Homing_Offset that makes that snapped correction permanent.

USAGE -- hold the motors you're calibrating at the fully straight/extended reference pose
(matching the hand's documented zero), then run, e.g.:

    python calibrate_motor_offsets.py --motors 2 10 11
    python calibrate_motor_offsets.py --motors 2 10 11 --dry-run   # preview only, no write
"""

import argparse
import sys

import dynamixel_sdk as dxl

PROTOCOL_VERSION = 2.0
BAUDRATE = 4000000

ADDR_TORQUE_ENABLE = 64
ADDR_HOMING_OFFSET = 20
ADDR_PRESENT_POSITION = 132

RESOLUTION = 4096  # ticks per revolution (XC330/XL330 and other X-series motors used here)
EXPECTED_STRAIGHT_TICKS = 2048  # pi radians = documented "fully open/straight" pose
STEP = RESOLUTION // 4  # 1024 ticks = 90 degrees -- the only increment a horn mount error occurs in


def _read_i32(packet, port, motor_id, address):
    val, result, error = packet.read4ByteTxRx(port, motor_id, address)
    if result != dxl.COMM_SUCCESS:
        raise RuntimeError(f"ID {motor_id}: read failed ({packet.getTxRxResult(result)})")
    if val > 0x7FFFFFFF:  # interpret as signed 32-bit
        val -= 0x100000000
    return val


def _write_i32(packet, port, motor_id, address, value):
    result, error = packet.write4ByteTxRx(port, motor_id, address, value & 0xFFFFFFFF)
    if result != dxl.COMM_SUCCESS:
        raise RuntimeError(f"ID {motor_id}: write failed ({packet.getTxRxResult(result)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--motors", type=int, nargs="+", required=True,
        help="Motor IDs to calibrate (hold these joints fully straight/extended before running).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without actually writing it.",
    )
    args = parser.parse_args()

    port = dxl.PortHandler(args.port)
    packet = dxl.PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        sys.exit(f"Failed to open {args.port}")
    if not port.setBaudRate(BAUDRATE):
        sys.exit("Failed to set baud rate")

    print(
        f"Calibrating motors {args.motors} -- make sure these joints are all currently held "
        f"at the fully straight/extended reference pose.\n"
    )

    try:
        for motor_id in args.motors:
            # Torque must be off to write Homing_Offset (EEPROM-area register on this model).
            _write_i32(packet, port, motor_id, ADDR_TORQUE_ENABLE, 0)

            current_offset = _read_i32(packet, port, motor_id, ADDR_HOMING_OFFSET)
            present_position = _read_i32(packet, port, motor_id, ADDR_PRESENT_POSITION)
            raw_sensor = present_position - current_offset  # true reading, independent of any prior calibration

            diff = raw_sensor - EXPECTED_STRAIGHT_TICKS
            snapped_diff = round(diff / STEP) * STEP
            new_offset = -snapped_diff

            print(
                f"ID {motor_id}: raw_sensor={raw_sensor}  diff_from_{EXPECTED_STRAIGHT_TICKS}={diff:+d} ticks "
                f"({diff / STEP * 90:+.1f} deg)  ->  snapped={snapped_diff:+d} ticks "
                f"({snapped_diff / STEP * 90:+.0f} deg)  ->  new Homing_Offset={new_offset:+d} "
                f"(was {current_offset:+d})"
            )

            if not args.dry_run:
                _write_i32(packet, port, motor_id, ADDR_HOMING_OFFSET, new_offset)
                verify_pos = _read_i32(packet, port, motor_id, ADDR_PRESENT_POSITION)
                print(f"           Present_Position now reads {verify_pos} (expect close to {EXPECTED_STRAIGHT_TICKS})")
    finally:
        port.closePort()

    if args.dry_run:
        print("\n(dry run -- nothing was written)")


if __name__ == "__main__":
    main()
