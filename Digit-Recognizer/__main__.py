import os
import sys
import shutil

# Grab the directory where __main__.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, ".idea"))

import mat