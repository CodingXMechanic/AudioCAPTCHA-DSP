#!/usr/bin/env python
"""Keep the machine awake for long benchmark runs (Windows).

Prevents the OS from suspending the system while a long experiment runs —
a sleep mid-run kills worker processes and truncates progress (this very
repo lost a run to it once). Uses ``SetThreadExecutionState`` (no admin
rights needed); optionally also flips the active power plan's AC standby
timeout when run with ``--set-power-cfg`` (needs elevation for some plans).

Usage
-----
    # hold the system awake until Ctrl+C (or parent process exits):
    python scripts/keep_awake.py

    # with a wall-clock limit (seconds), e.g. 8 hours:
    python scripts/keep_awake.py --seconds 28800

    # also try to disable AC standby in the power plan (best effort):
    python scripts/keep_awake.py --set-power-cfg

``SetThreadExecutionState`` flags apply per calling *thread* and persist for
the process lifetime while ``ES_CONTINUOUS`` remains set; the script keeps a
handle alive for the requested duration and clears the state on exit.
"""
from __future__ import annotations

import argparse
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # optional: also keep the screen on


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--seconds", type=float, default=0.0,
                   help="hold for this many seconds (0 = until killed)")
    p.add_argument("--display", action="store_true",
                   help="also keep the display on")
    p.add_argument("--set-power-cfg", action="store_true",
                   help="best-effort: disable AC standby in the power plan")
    args = p.parse_args(argv)

    if args.set_power_cfg:
        import subprocess

        for cmd in (
            ["powercfg", "/change", "standby-timeout-ac", "0"],
            ["powercfg", "/change", "hibernate-timeout-ac", "0"],
        ):
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
            except Exception:
                pass

    try:
        import ctypes
    except ImportError:
        print("[keep_awake] not Windows — nothing to do", file=sys.stderr)
        return 0

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if args.display:
        flags |= ES_DISPLAY_REQUIRED
    prev = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if prev == 0:
        print("[keep_awake] SetThreadExecutionState failed", file=sys.stderr)
        return 1
    print(f"[keep_awake] holding system awake (flags=0x{flags:08X})"
          + (f" for {args.seconds:g}s" if args.seconds else " until killed"))
    try:
        t0 = time.time()
        while True:
            time.sleep(30)
            # re-assert periodically (cheap; guards against state resets)
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
            if args.seconds and time.time() - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("[keep_awake] released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
