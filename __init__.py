import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Base module should export 512 and 1024 DPR classes
from transform_1024 import DPR_1024
from transform_512 import DPR_512
