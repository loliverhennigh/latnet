#!/usr/bin/env python

import sys

# import latnet librarys
sys.path.append('../latnet')
from controller import LatNetController

# import important librarys
import numpy as np
import glob

if __name__ == '__main__':
  network_ctrl = LatNetController(sailfish_sim='./2d_flow_generator.py')
  network_ctrl.train()
    

