import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

model = mujoco.MjModel.from_xml_path("./simulation/cartpole.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=224, width=224)

def render_state(x, theta, fname):
    data.qpos[0] = x
    data.qpos[1] = theta
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="side")
    img = renderer.render()
    Image.fromarray(img).save(fname)
    print(f"saved {fname}  (x={x}, theta={theta:.2f})")

# A spread of cart positions at rest (theta=0, hanging down)
render_state(0.0, 0.0, "./simulation/frame_center_down.png")
render_state(-0.9, 0.0, "./simulation/frame_left_down.png")
render_state(0.9, 0.0, "./simulation/frame_right_down.png")

# Upright, centered -- candidate "goal image" look
render_state(0.0, np.pi, "./simulation/frame_center_up.png")

# Mid-swing at a few angles, centered
render_state(0.0, np.pi/2, "./simulation/frame_center_mid.png")
render_state(0.0, -np.pi/4, "./simulation/frame_center_tilt.png")