"""Print a formatted john-linux hardware status snapshot.

This script is meant for SSH monitoring.  It uses psutil for CPU, memory,
disk, and sensors, and NVML when the NVIDIA driver is available.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Iterable

import psutil
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def _format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def _color_temp(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    temp = float(value)
    text = f"{temp:.1f}C"
    if temp < 45:
        color = Fore.CYAN
    elif temp < 60:
        color = Fore.GREEN
    elif temp < 75:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    return color + text + Style.RESET_ALL


def _color_percent(value: float, warn: float = 70.0, danger: float = 90.0) -> str:
    text = f"{value:.1f}%"
    if value >= danger:
        color = Fore.RED
    elif value >= warn:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN
    return color + text + Style.RESET_ALL


def _sensor_rows(limit: int) -> list[list[str]]:
    temps = psutil.sensors_temperatures(fahrenheit=False) or {}
    rows: list[tuple[str, str, float | None, float | None]] = []
    for chip, entries in temps.items():
        for entry in entries:
            rows.append((chip, entry.label or "-", entry.current, entry.high))

    def sort_key(row: tuple[str, str, float | None, float | None]) -> float:
        return float("-inf") if row[2] is None else row[2]

    table_rows = []
    for chip, label, current, high in sorted(rows, key=sort_key, reverse=True)[:limit]:
        high_s = "n/a" if high is None else f"{high:.1f}C"
        table_rows.append([chip, label, _color_temp(current), high_s])
    return table_rows


def _gpu_rows() -> tuple[list[list[str]], str | None]:
    try:
        import pynvml
    except Exception as exc:  # noqa: BLE001
        return [], f"pynvml unavailable ({exc})"

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return [], f"NVML unavailable ({exc})"

    rows: list[list[str]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return [], "no NVIDIA device reported by NVML"
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except Exception:  # noqa: BLE001
                pass
            power_s = "n/a" if power is None else f"{power:.1f}W"
            rows.append(
                [
                    f"gpu{idx}",
                    str(name),
                    _color_temp(temp),
                    f"{util.gpu}%",
                    f"{_format_bytes(mem.used)} / {_format_bytes(mem.total)}",
                    power_s,
                ]
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return rows, None


def snapshot(sensor_limit: int) -> str:
    cpu_percent = psutil.cpu_percent(interval=0.2)
    load = psutil.getloadavg()
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")

    summary_rows = [
        ["CPU Usage", _color_percent(cpu_percent)],
        ["Load Avg", f"{load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}"],
        [
            "Memory",
            f"{_color_percent(mem.percent)}  "
            f"{_format_bytes(mem.used)} / {_format_bytes(mem.total)}",
        ],
        [
            "Swap",
            f"{_color_percent(swap.percent)}  "
            f"{_format_bytes(swap.used)} / {_format_bytes(swap.total)}",
        ],
        [
            "Disk /",
            f"{_color_percent(disk.percent, warn=80.0, danger=92.0)}  "
            f"{_format_bytes(disk.used)} / {_format_bytes(disk.total)}",
        ],
    ]

    lines = [Fore.MAGENTA + datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    lines.append("")
    lines.append(Fore.CYAN + "=== System Summary ===" + Style.RESET_ALL)
    lines.append(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    sensor_rows = _sensor_rows(sensor_limit)
    lines.append("")
    lines.append(Fore.CYAN + "=== Temperature Sensors ===" + Style.RESET_ALL)
    if sensor_rows:
        lines.append(
            tabulate(
                sensor_rows,
                headers=["Chip", "Sensor", "Temp", "High"],
                tablefmt="fancy_grid",
            )
        )
    else:
        lines.append("No temperature sensors reported by psutil.")

    gpu_rows, gpu_error = _gpu_rows()
    lines.append("")
    lines.append(Fore.CYAN + "=== GPU ===" + Style.RESET_ALL)
    if gpu_rows:
        lines.append(
            tabulate(
                gpu_rows,
                headers=["ID", "Name", "Temp", "Util", "Memory", "Power"],
                tablefmt="fancy_grid",
            )
        )
    else:
        lines.append(gpu_error or "No GPU data available.")
    return "\n".join(lines)


def _run_watch(interval: float, sensor_limit: int) -> None:
    while True:
        print(snapshot(sensor_limit), flush=True)
        print("", flush=True)
        time.sleep(interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="print snapshots repeatedly",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="watch interval in seconds",
    )
    parser.add_argument(
        "--sensor-limit",
        type=int,
        default=12,
        help="maximum temperature sensor lines to print",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.watch:
        _run_watch(max(0.5, args.interval), args.sensor_limit)
    else:
        print(snapshot(args.sensor_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
