import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from controller import CartpoleController

def test_controller(test_mode, duration=None):
    model = mujoco.MjModel.from_xml_path('./simulation/cartpole.xml')
    data = mujoco.MjData(model)
    
    # Physics is 500Hz (0.002s), Control is 20Hz (0.05s)
    control_rate = 20
    physics_rate = int(1.0 / model.opt.timestep)
    substeps = physics_rate // control_rate
    
    controller = CartpoleController(model, data)
    
    if test_mode == 'lqr':
        # Start near upright
        data.qpos[0] = 0.0
        data.qpos[1] = np.pi + 0.25 # Slight perturbation
        controller.mode = 1 # Force LQR mode
    elif test_mode == 'swingup':
        # Start at bottom
        data.qpos[0] = 0.0
        data.qpos[1] = 0.01 # Tiny initial angle to break perfect symmetry
        controller.mode = 0
    else:
        raise ValueError(f"Unknown test mode: {test_mode}")
        
    mujoco.mj_forward(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            if duration is not None and (step_start - start_time) > duration:
                break
            
            # 1. Get control action (at 20Hz)
            action = controller.get_action()
            
            # Apply actuator limits (e.g. [-10, 10]) from xml
            ctrlrange = model.actuator_ctrlrange[0]
            if model.actuator_ctrllimited[0]:
                action = np.clip(action, ctrlrange[0], ctrlrange[1])
                
            data.ctrl[0] = action
            
            # 2. Step physics multiple times holding action constant
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                
            # 3. Update viewer
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.sync()
            
            # 4. Try to run in real-time
            time_until_next_step = model.opt.timestep * substeps - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['lqr', 'swingup'], required=True, help='Test mode')
    parser.add_argument('--duration', type=float, default=None, help='Auto-exit after duration (seconds)')
    args = parser.parse_args()
    
    test_controller(args.test, args.duration)
