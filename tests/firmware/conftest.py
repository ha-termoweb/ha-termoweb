"""Build firmware/radio_core with the host gcc into a ctypes-loadable library."""

from __future__ import annotations

import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "firmware" / "radio_core"
MAKEFILE = ROOT / "firmware" / "nanocul" / "Makefile"
HOST_HAL = Path(__file__).with_name("host_hal.c")


def nanocul_make_var(name: str) -> str:
    """Return a variable's value from the nanoCUL Makefile."""
    match = re.search(rf"^{name}\s*=\s*(\S+)", MAKEFILE.read_text(), re.MULTILINE)
    assert match is not None, name
    return match.group(1)


class Core:
    """Thin wrapper over the compiled core plus the fake-chip host HAL."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        """Declare the C signatures used by the tests."""
        self.lib = lib
        u8, u16, u32 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32
        buf = ctypes.c_char_p
        sigs = {
            "rc_crc_ccitt": (u16, [buf, u16]),
            "rc_crc_modbus": (u16, [buf, u16]),
            "rc_frame_total": (u16, [u8, u8]),
            "rc_frame_crc_ok": (u8, [u8, buf, u16]),
            "rc_field": (u8, [u8, buf, u8]),
            "rc_build_ack": (None, [u8, u16, u8, u8, buf]),
            "rc_freq_word": (u32, [u32]),
            "rc_rssi_dbm10": (ctypes.c_int16, [u8]),
            "rc_self_test": (u8, []),
            "rc_begin": (None, []),
            "rc_feed": (u8, [ctypes.c_char]),
            "rc_poll": (None, []),
            "rc_service": (None, []),
            "rc_on_gdo2": (None, []),
            "rc_on_gdo0": (None, [u32]),
            "rc_print_survey_on": (None, [u16, ctypes.c_int16, ctypes.c_int16]),
            "rc_print_survey_off": (None, [u16]),
            "rc_print_rawb": (None, [u16, ctypes.c_int16]),
            "rc_print_raw_start": (None, [u16]),
            "rc_print_raw_run": (None, [u8, u16]),
            "rc_print_rawe": (None, [u16, u16]),
            "rc_print_eol": (None, []),
            "rc_print_overrun": (None, []),
            "host_reset": (None, []),
            "host_take_output": (u32, [buf, u32]),
            "host_advance": (None, [u32]),
            "host_rx_push": (None, [buf, u32]),
            "host_set_rx_overflow": (None, [u8]),
            "host_set_status": (None, [u8, u8]),
            "host_set_absent": (None, [ctypes.c_int]),
            "host_radio_ok": (u8, []),
            "host_set_marcstate_tx": (None, [u8]),
            "host_reg": (u8, [u8]),
            "host_strobes": (u32, [u8]),
            "host_tx_count": (u32, []),
            "host_tx_take": (u32, [buf]),
            "host_masked": (ctypes.c_int, []),
            "host_lock_depth": (ctypes.c_int, []),
            "host_survey_calls": (ctypes.c_int, []),
            "host_last_survey_secs": (ctypes.c_int, []),
            "host_dialect": (u8, []),
            "host_auto_ack": (u8, []),
            "host_our_id": (u8, []),
            "host_net": (u16, []),
            "host_rx_state": (u8, []),
            "host_pstate": (u8, []),
            "host_frames_ok": (u32, []),
            "host_freq_khz": (u32, []),
            "host_set_mac": (None, [buf]),
            "host_set_survey_active": (None, [u8]),
            "host_clear_packet_ready": (None, []),
        }
        for name, (restype, argtypes) in sigs.items():
            func = getattr(lib, name)
            func.restype = restype
            func.argtypes = argtypes
        self._mac = b""

    def __getattr__(self, name: str):
        """Forward anything else to the library."""
        return getattr(self.lib, name)

    def output(self) -> str:
        """Return and clear everything the firmware printed."""
        buf = ctypes.create_string_buffer(1 << 16)
        n = self.lib.host_take_output(buf, len(buf))
        return buf.raw[:n].decode("ascii")

    def lines(self) -> list[str]:
        """Return and clear the printed lines; every one must end CR LF."""
        text = self.output()
        assert text == "" or text.endswith("\r\n"), text
        return text.split("\r\n")[:-1] if text else []

    def send(self, text: str) -> list[str]:
        """Feed ``text`` byte by byte and return the lines it printed."""
        for char in text.encode("ascii"):
            self.lib.rc_feed(bytes([char]))
        return self.lines()

    def start(
        self, mac: bytes | None = None, absent_level: int | None = None
    ) -> list[str]:
        """Reset the fake chip, boot the core and return the boot lines.

        ``absent_level`` (0x00 or 0xFF) boots with no chip on the bus: every
        status read returns that MISO idle level.
        """
        self.lib.host_reset()
        self._mac = mac or b""  # keep the bytes alive while C points at them
        if mac is not None:
            self.lib.host_set_mac(self._mac)
        if absent_level is not None:
            self.lib.host_set_absent(absent_level)
        self.lib.rc_begin()
        return self.lines()

    def tx_frame(self) -> bytes:
        """Return the bytes the last transmit wrote to the TX FIFO."""
        buf = ctypes.create_string_buffer(512)
        n = self.lib.host_tx_take(buf)
        return buf.raw[:n]

    def receive(
        self, air: bytes, status: bytes = b"\x2c\x95", edge_us: int = 5000
    ) -> None:
        """Deliver one on-air frame through GDO2/GDO0 as the chip would."""
        self.lib.host_rx_push(air + status, len(air) + len(status))
        self.lib.rc_on_gdo2()
        self.lib.rc_on_gdo0(edge_us)
        self.lib.rc_service()


@pytest.fixture(scope="session")
def core_lib(tmp_path_factory: pytest.TempPathFactory) -> Core:
    """Compile the core with gcc (skip when there is none) and load it."""
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    lib = tmp_path_factory.mktemp("radio_core") / "libradiocore.so"
    version = nanocul_make_var("VERSION")
    pa = nanocul_make_var("PA")
    subprocess.run(
        [
            gcc,
            "-shared",
            "-fPIC",
            "-O1",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            f'-DRC_VERSION="{version}"',
            f"-DRC_PA={pa}",
            f"-I{CORE}",
            str(CORE / "radio_core.c"),
            str(CORE / "cc1101.c"),
            str(HOST_HAL),
            "-o",
            str(lib),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Core(ctypes.CDLL(str(lib)))


@pytest.fixture
def core(core_lib: Core) -> Core:
    """Return the core after a fresh boot (its banner already consumed)."""
    core_lib.start()
    return core_lib
