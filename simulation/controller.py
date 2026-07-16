import numpy as np
import scipy.linalg
import mujoco

class CartpoleController:
    def __init__(self, model, data, 
                 k_swing=100.0, k_cart=0.0, 
                 theta_sw=np.deg2rad(20), omega_sw=2.0, 
                 theta_sw_out=np.deg2rad(35)):
        self.model = model
        self.data = data
        
        self.k_swing = k_swing
        self.k_cart = k_cart
        self.theta_sw = theta_sw
        self.omega_sw = omega_sw
        self.theta_sw_out = theta_sw_out
        
        # State: 0=Swing-up, 1=LQR
        self.mode = 0 
        
        # Exact parameters from model for energy calc
        pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self.m = self.model.body_mass[pole_id]
        self.l = np.linalg.norm(self.model.body_ipos[pole_id])
        # Inertia around hinge (Y-axis) using parallel axis theorem
        self.I = self.model.body_inertia[pole_id][1] + self.m * (self.l**2)
        self.g = abs(self.model.opt.gravity[2])
        self.E_top = 2 * self.m * self.g * self.l
        
        # Calculate LQR gain
        self.K = self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        """Linearize about the upright equilibrium and compute continuous LQR gain K."""
        # Save current state and options
        qpos_orig = self.data.qpos.copy()
        qvel_orig = self.data.qvel.copy()
        qacc_orig = self.data.qacc.copy()
        ctrl_orig = self.data.ctrl.copy()
        integrator_orig = self.model.opt.integrator
        
        nv = self.model.nv
        nu = self.model.nu
        
        # Set to upright equilibrium
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = np.pi
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # Use Euler integrator for straightforward discrete -> continuous conversion
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        mujoco.mj_forward(self.model, self.data)
        
        # Allocate discrete matrices
        A_d = np.zeros((2*nv, 2*nv))
        B_d = np.zeros((2*nv, nu))
        
        # Numerically linearize exact model via mjd_transitionFD
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, A_d, B_d, None, None)
        
        # Convert discrete Euler transition matrices to continuous-time A and B
        h = self.model.opt.timestep
        A = (A_d - np.eye(2*nv)) / h
        B = B_d / h
        
        # Restore original state and options
        self.data.qpos[:] = qpos_orig
        self.data.qvel[:] = qvel_orig
        self.data.qacc[:] = qacc_orig
        self.data.ctrl[:] = ctrl_orig
        self.model.opt.integrator = integrator_orig
        mujoco.mj_forward(self.model, self.data)
        
        # LQR Cost Matrices
        Q = np.diag([10.0, 10.0, 1.0, 1.0])  # Penalize position and angle more
        R = np.array([[0.1]])
        
        # Solve continuous algebraic Riccati equation
        # A.T P + P A - P B R^-1 B.T P + Q = 0
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        
        # K = R^-1 B.T P
        K = np.linalg.inv(R) @ B.T @ P
        
        return K

    def get_action(self):
        # Current state
        x = self.data.qpos[0]
        theta = self.data.qpos[1]
        x_dot = self.data.qvel[0]
        theta_dot = self.data.qvel[1]
        
        # Deviation from upright (wrapped to [-pi, pi])
        theta_dev = (theta - np.pi + np.pi) % (2 * np.pi) - np.pi
        
        # Switching logic
        if self.mode == 0:  # Swing-up
            if abs(theta_dev) < self.theta_sw and abs(theta_dot) < self.omega_sw:
                self.mode = 1  # Switch to LQR
        else:  # LQR
            if abs(theta_dev) > self.theta_sw_out:
                self.mode = 0  # Switch to Swing-up
                
        if self.mode == 1:
            # LQR control
            s = np.array([x, theta_dev, x_dot, theta_dot])
            u = -self.K @ s
            return u[0]
        else:
            # Swing-up control (Åström–Furuta)
            # Energy E = 1/2 m l^2 theta_dot^2 + m g l (1 - cos(theta))
            # Note: in standard formula, theta=0 is upright. Here theta=0 is down.
            # So cos(theta) is 1 when down, -1 when up.
            # Upright energy: E_top = 2 m g l
            # Current energy:
            # With theta=0 down, the height is l * (1 - cos(theta)) 
            # Wait, 1 - cos(0) = 0 (bottom). 1 - cos(pi) = 2 (top). Correct.
            E = 0.5 * self.I * (theta_dot**2) + self.m * self.g * self.l * (1 - np.cos(theta))
            
            # Control law: a = k * (E - E_top) * sign(theta_dot * cos(theta))
            # In our coordinate system, when swinging past bottom (theta=0), cos(0) = 1.
            # If we want to pump energy, we push in direction of theta_dot * cos(theta).
            a_cmd = self.k_swing * (self.E_top - E) * np.sign(theta_dot * np.cos(theta))
            
            # Add proportional term to pull cart to center
            cart_correction = -self.k_cart * x
            
            return a_cmd + cart_correction
