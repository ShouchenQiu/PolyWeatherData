"""CLI entry point for PolyWeatherData.

Usage
-----
    streamlit run ui/app.py
    python -m scripts.run_mispricing_analysis
"""
from __future__ import annotations

import sys


def main() -> None:
    print("PolyWeatherData -- Polymarket weather mispricing analysis")
    print()
    print("Usage:")
    print("  streamlit run ui/app.py         # Launch analysis dashboard")
    print("  python -m scripts.run_mispricing_analysis  # Batch mispricing scan")
    sys.exit(0)


if __name__ == "__main__":
    main()
