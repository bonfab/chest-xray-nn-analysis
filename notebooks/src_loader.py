#!/usr/bin/env python3

import sys
import os

module_path = os.path.abspath(os.path.join(os.pardir, "src"))
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path = os.path.abspath(os.path.join(os.pardir, "LRP"))
if module_path not in sys.path:
    sys.path.append(module_path)
