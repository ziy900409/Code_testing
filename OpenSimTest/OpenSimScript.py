# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:02:19 2023

1. https://simtk-confluence.stanford.edu:8443/display/OpenSim/Command+Line+Utilities

操作步驟 :

1. Navigate to specific folder.
2. executing opensim-cmd run tool.
    2.1. Scaling model (subject_Setup_Scaling.xml)
        2.1.1. input : simbody_model (.osim), static motion (.trc)
        2.1.2. output : subject_simbody (.osim)
    2.2. Inverse Kinematics (subject_Setup_IK.xml)
        2.2.1. input : subject_simbody (.osim), dynamic motion (.trc) 
        2.2.2. output : subject_dynamic_ik (.mot)
    2.3. Inverse Dynamics (subject_Setup_InverseDynamics.xml )
        2.3.1. input : subject_simbody (.osim), subject_dynamic_ik (.mot), external force (.xml)
        2.3.2. output : subject_dynamic_InverseDynamics_force (.sto)
    2.4. Static Optimization
        2.4.1. input : subject_simbody (.osim)
    2.5. Computed Muscle Control

@author: Hsin.YH.Yang
"""

import subprocess
import shlex
# subprocess.Popen("echo Hello World", shell=True, stdout=subprocess.PIPE).stdout.read()

process = subprocess.run(["date"], check=True)
print(process)
# 調用 subprocess 開啟 OpenSim
process1 = subprocess.Popen("opensim-cmd")
print(process1)