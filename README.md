# ARPOD-Sims-Python

 Running ARPOD Simulations in Python 
 Author: Anthony Aborizk
 Date: 8/7/2023

 This repository holds several variations of the ARPOD (Autonomous rendezvous, proximity operations, and docking) problem dynamics. Each variation is refered to as an "environment" and is coded in Python following the OpenAI gymnasium structure. Please run the following line of code in a Python3.9 environment before attempting to run any files: 
 
    pip install -r requirements.txt 

 All enviromnents included herein are represented in Hill's (CWH) reference frame. The following assumptions are made on the dynamics of the ARPOD systems: 

- The target is in a circular orbit about the Earth
- The origin of the CWH frame is located at the center of the Target
- The body-fixed frame of the deputy is aligned along the principal axes of the deputy 
- The body-fixed frame of the deputy is aligned with the CWH frame when its orientation is [0,0,0]
- All controller thrusters apply a force through the center of mass of the deputy
- All controller flywheels rotate about the principal axes of the deputy
 
The user can decide between the following simulation options: 

 1. Fully-actuated ARPOD in 3D space\
    a. The chaser can translate about all three axes but cannot rotate

 2. Fully-actuated ARPOD with a 6-DOF model\ (This environment is still under development)
    a. This includes translation and rotation about all three axes. Translation is modeled using the CWH equations and rotation is modeled using Modified Rodrigues Parameters (MRPs).

 3. Under-actuated ARPOD with a 3-DOF planar model\
    a. This includes translation about the local vertical and local horizontal axes as
    well as rotation about the angular momentum vector\
    b. This simulation assumes that the chaser is under-actuated (i.e. the chaser can only control a flywheel that is aligned with the angular momentum vector and a thruster that is aligned with the horizontal axis of the deputy)

 4. Under-actuated ARPOD with a 6-DOF model\
    a. This includes translation and rotation about all three axes\
    b. This simulation assumes that the chaser is under-actuated (i.e. the chaser can
       control all flywheels and a single thruster that is aligned with the horizontal axis)

# Repository Structure
```Thehierarchy of the repository is as follows:
.
├── environments
│   ├── actuated3dof.py       # Fully-actuated ARPOD in 3D space
│   ├── actuated6dof.py       # Fully-actuated ARPOD with a 6-DOF model
│   ├── underactuated6dof.py  # Under-actuated ARPOD with a 6-DOF model
|   ├── unitls.py             # Utility functions for the ARPOD environments
|   ├── rendering.py          # Rendering functions for the planar ARPOD environments
│   └── __init__.py
.
|── documents                 # Contains relevent ARPOD papers and supporting documents
. 
|── Running_Things.py         # Runs the ARPOD simulations with a dummy controller
```
