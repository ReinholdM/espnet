"""Initialize sub package."""
import sys
import os
if "/home/espnet" in sys.path:
    sys.path.remove("/home/espnet")
ESPNET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.insert(0, ESPNET_ROOT)
