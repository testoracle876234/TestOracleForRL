import Util
import LPEA_Env
import numpy as np
from gymnasium.wrappers import TimeLimit 
import stable_baselines3 as sb3
import Lyaponov_oracle_util as LO
import os
import sys

bug_no = int(sys.argv[1])
algorithm = sys.argv[2]
n = int(sys.argv[3])
m = int(sys.argv[4])
I = int(sys.argv[5])
J = int(sys.argv[6])

buggy_metrics = LO.buggy_trained_model_metrics_calculation(algorithm, n, m, I, J, bug_no)

for vartheta in range(100, 40, -10):
    for theta in range(100, 45, -25):
        Oracle_result = LO.LPEA_Oracle(buggy_metrics, I, J, vartheta * 0.01, theta * 0.01)
        if Oracle_result:
            print("vartheta={vartheta}%, theta={theta}%, the software is bug-less based on LPEA Oracle".format(vartheta=vartheta, theta=theta))
        else:
            print("vartheta={vartheta}%, theta={theta}%, the software is buggy based on LPEA Oracle".format(vartheta=vartheta, theta=theta))
