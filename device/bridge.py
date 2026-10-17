"""Send the TwinRead status to the UNO Q MCU (LED matrix) over the Arduino Bridge RPC. Never raises.

ASSUMPTIONS (unverified on a board; must be checked on the UNO Q):
  1. Inside Arduino App Lab, Python apps can `from arduino.app_utils import Bridge` and call `Bridge.call("set_state", s)`.
  2. Otherwise the Linux-side `arduino-router` listens on a unix socket (assumed /var/run/arduino-router.sock) and speaks
     MessagePack-RPC: request = [0, msgid, "method", [params]]. A tiny encoder for that shape is below (stdlib only).
The MCU sketch registers the method "set_state" (device/mcu/twinread_led). Off the board both paths are absent and
send_status returns False (a no-op), so the app keeps running without the LED.
"""
from __future__ import annotations

import os
import socket
import struct

STATES = ("idle", "busy", "agree", "review", "no_tissue", "error")
METHOD = "set_state"
ROUTER_SOCK = os.environ.get("ARDUINO_ROUTER_SOCK", "/var/run/arduino-router.sock")
_last_state = None


def _app_utils_bridge():
    try:
        from arduino.app_utils import Bridge  # only exists inside App Lab on the board
        return Bridge
    except Exception:
        return None


def _mp_str(s: str) -> bytes:
    b = s.encode()
    return (bytes([0xA0 | len(b)]) if len(b) < 32 else b"\xd9" + bytes([len(b)])) + b


def _mp_request(msgid: int, method: str, param: str) -> bytes:
    # [0, msgid, method, [param]] as a msgpack fixarray(4); msgid < 128 here
    return b"\x94\x00" + (bytes([msgid]) if msgid < 128 else b"\xcd" + struct.pack(">H", msgid)) \
        + _mp_str(method) + b"\x91" + _mp_str(param)


def send_status(state) -> bool:
    """True if the MCU call was delivered, False if the Bridge is unavailable or the call failed."""
    global _last_state
    if state not in STATES:
        return False
    try:
        br = _app_utils_bridge()
        if br is not None:
            br.call(METHOD, state)
            _last_state = state
            return True
    except Exception:
        pass
    try:
        if os.path.exists(ROUTER_SOCK):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(ROUTER_SOCK)
                s.sendall(_mp_request(1, METHOD, state))
                resp = s.recv(256)  # [1, msgid, error, result]; error nil (0xc0) means success
                ok = len(resp) >= 4 and resp[0] == 0x94 and resp[1] == 0x01 and resp[2] == 1 and resp[3] == 0xC0
                _last_state = state if ok else _last_state
                return bool(ok)
    except Exception:
        pass
    return False
