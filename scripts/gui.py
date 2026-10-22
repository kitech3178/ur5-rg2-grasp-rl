"""Desktop GUI: algorithm / episodes / reward settings, live wrist camera,
reward test on single episodes, result graphs and comparison.

    python scripts/gui.py
"""

import sys

from _common import ROOT  # noqa: F401 - puts the project on sys.path

from grasp_rl.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
