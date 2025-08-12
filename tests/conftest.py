# -*- coding: utf-8 -*-

import os
import sys


# Ensure local package is imported instead of any site-packages installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


