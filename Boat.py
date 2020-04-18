import numpy as np
from rotations import *

global rho
rho = 1005 # Density of sea water, kg/m^3
global g
g = 9.806 # Gravitational acceleration, m/s^2


class Boat:
    '''
    Boat objects can move in time given external forces (waves and currents) and
    internal forces(propulsion and steering).

    The vessel is assumed to be a rectangular prism for mass and wave force
    calculations.

    The coefficients for inertia, drag forces, center of mass, and maneuvering
    surfaces can be defined below.
    '''

    def __init__(self, pos, v_self, theta, w, t = 0):
        '''
        pos:    numpy array of length 3 for position (x, y, z) in meters
        v_self: numpy array of length 3 for the vessel velocity in its own frame
                of reference (v_forward, v_sway, v_heave) in meters/s
        theta:  numpy array of length 3 for angles of rotation (roll, pitch,
                yaw) in radians
        w:      numpy array of length 3 for angular velocity (roll rate,
                pitch rate, yaw rate) in radians/s
        t:      float for current time on vessel in s. Used for syncing with
                waves
        '''
        # State parameters
        self.pos = pos
        self.v_self = v_self
        self.v_current_self = np.zeros(3)
        self.v = np.matmul(rotation_all(theta), self.v_self)
        self.theta = theta
        self.w = w
        self.t = t

        # Geometry
        self.L = 3 # Length, m
        self.W = 1 # Beam, m
        self.D = 0.5 # Draft, m
        self.cx = self.L/2 # Center of mass from stern, m

        # Grid bottom
        self.x_spacing = np.linspace(0, self.L, 11)
        self.y_spacing = np.linspace(-self.W/2, self.W/2, 11)
        X, Y = np.meshgrid(self.x_spacing, self.y_spacing)
        self.X_b = np.ravel(X);
        self.X_b = self.X_b - self.cx
        self.Y_b = np.ravel(Y);
        self.Z_b = -self.D * np.ones(np.shape(self.X_b))
        self.grid_b = np.array([self.X_b, self.Y_b, self.Z_b])
        self.grid_global_b = np.matmul(rotation_all(self.theta), self.grid_b) + np.tile(self.pos, (len(self.X_b), 1)).T

        # Grid left
        self.x_spacing = np.linspace(0, self.L, 11)
        self.z_spacing = np.linspace(-self.D, 0, 6)
        X, Z = np.meshgrid(self.x_spacing, self.z_spacing)
        self.X_l = np.ravel(X);
        self.X_l = self.X_l - self.cx
        self.Z_l = np.ravel(Z);
        self.Y_l = -self.W/2 * np.ones(np.shape(self.X_l))
        self.grid_l = np.array([self.X_l, self.Y_l, self.Z_l])
        self.grid_global_l = np.matmul(rotation_all(self.theta), self.grid_l) + np.tile(self.pos, (len(self.X_l), 1)).T

        # Grid right
        self.X_r = self.X_l
        self.Z_r = self.Z_l;
        self.Y_r = self.W/2 * np.ones(np.shape(self.X_l))
        self.grid_r = np.array([self.X_r, self.Y_r, self.Z_r])
        self.grid_global_r = np.matmul(rotation_all(self.theta), self.grid_r) + np.tile(self.pos, (len(self.X_r), 1)).T

        # Grid front
        self.y_spacing = np.linspace(-self.W/2, self.W/2, 11)
        self.z_spacing = np.linspace(-self.D, 0, 6)
        Y, Z = np.meshgrid(self.y_spacing, self.z_spacing)
        self.Y_f = np.ravel(Y)
        self.Z_f = np.ravel(Z);
        self.X_f = (self.L-self.cx) * np.ones(np.shape(self.Y_f))
        self.grid_f = np.array([self.X_f, self.Y_f, self.Z_f])
        self.grid_global_f = np.matmul(rotation_all(self.theta), self.grid_f) + np.tile(self.pos, (len(self.X_f), 1)).T

        # Grid stern
        self.Y_s = self.Y_f
        self.Z_s = self.Z_f
        self.X_s = -self.cx * np.ones(np.shape(self.Y_s))
        self.grid_s = np.array([self.X_s, self.Y_s, self.Z_s])
        self.grid_global_s = np.matmul(rotation_all(self.theta), self.grid_s) + np.tile(self.pos, (len(self.X_s), 1)).T

        # Areas
        self.Az = self.L * self.W
        self.Ax = self.D * self.W
        self.Ay = self.L * self.D

        # Mass
        self.M = rho * self.W*self.L*self.D

        # Moment of inertia
        self.I = np.array([50, 70, 500])

        # Propulsion
        self.n_prop = 1 # Number of propellers
        self.d_prop = 0.2 # Propeller diameter, m
        self.v_prop = 0 # Propeller speed, rad/s
        self.K_T = 1 # Thrust coefficient

        # Heave
        self.b_h = 1
        self.K_h = 1
        # Pitch
        self.b_p = 1
        self.K_p = 1
        # Roll
        self.b_r = 1
        self.K_r = 1
        # Yaw and Rudder
        self.A_rd = 0.5 # Rudder area, m^2
        self.K_rd = 0.065 # Angular lift coefficient of rudder, 1/deg
        self.t_rd = 0 # Rudder angle, deg
        # Drag
        self.C_df = 0.1 # Forward drag coefficient
        self.C_dl = 5.1 # Lateral drag coefficient
        self.C_dw = np.array([5, 10, 5]) # Angular drag coefficients


    def set_grid(self):
        self.grid_global_b = np.matmul(rotation_all(self.theta), self.grid_b) + np.tile(self.pos, (len(self.X_b), 1)).T
        self.grid_global_l = np.matmul(rotation_all(self.theta), self.grid_l) + np.tile(self.pos, (len(self.X_l), 1)).T
        self.grid_global_r = np.matmul(rotation_all(self.theta), self.grid_r) + np.tile(self.pos, (len(self.X_r), 1)).T
        self.grid_global_f = np.matmul(rotation_all(self.theta), self.grid_f) + np.tile(self.pos, (len(self.X_f), 1)).T
        self.grid_global_s = np.matmul(rotation_all(self.theta), self.grid_s) + np.tile(self.pos, (len(self.X_s), 1)).T

    # Heave force
    def F_h(self, wave):
        if wave == None: return 0
        height = lambda x, y: wave.height(x, y, self.t)
        H = np.mean(height(self.grid_global_b[0], self.grid_global_b[1]), axis = len(np.shape(self.t)))
        F = (H * self.K_h * self.Az) - (self.b_h * self.M * self.v_self[2])
        return F

    # Roll torque
    def T_r(self, wave, t):
        return 0

    # Pitch torque
    def T_p(self, wave, t):
        return 0

    # Yaw torque
    def T_y(self):
        CL = self.K_rd * self.t_rd
        F_rudder = 0.5 * rho * self.v_self[0]*abs(self.v_self[0]) * CL * self.A_rd
        torque = F_rudder * (-self.cx + self.L)
        return torque

    # Angular drag
    def D_w(self):
        return 0.5 * rho * self.w*np.abs(self.w) * self.C_dw

    # Propulsive force
    def F_prop(self):
        F = self.n_prop * self.K_T * self.v_prop*abs(self.v_prop) * self.d_prop**4
        return F

    # Forward drag
    def D_forward(self):
        v = self.v_self[0] - self.v_current_self[0]
        F = 0.5 * rho * v*abs(v) * self.C_df * (self.Ax + self.Az)
        return F

    # Lateral drag
    def D_lateral(self):
        v = self.v_self[1] - self.v_current_self[1]
        F = 0.5 * rho * v*abs(v) * self.C_dl * (self.Ay + self.Az)
        return F

    # Wave forces
    def W_forward(self, wave):
        if wave == None: return 0
        pressure = lambda x, y, z: wave.pressure(x, y, z, self.t)
        P_front = np.mean(pressure(self.grid_global_f[0], self.grid_global_f[1], self.grid_global_f[2]), axis = len(np.shape(self.t)))
        P_stern = np.mean(pressure(self.grid_global_s[0], self.grid_global_s[1], self.grid_global_s[2]), axis = len(np.shape(self.t)))
        F = (P_stern - P_front) * self.Ax
        return F

    def W_lateral(self, wave):
        if wave == None: return 0
        pressure = lambda x, y, z: wave.pressure(x, y, z, self.t)
        P_left = np.mean(pressure(self.grid_global_l[0], self.grid_global_l[1], self.grid_global_l[2]), axis = len(np.shape(self.t)))
        P_right = np.mean(pressure(self.grid_global_r[0], self.grid_global_r[1], self.grid_global_r[2]), axis = len(np.shape(self.t)))
        F = (P_right - P_left) * self.Ay
        return F

    # Set rudder
    def rudder_to(self, t_rd):
        self.t_rd = t_rd

    # Set propeller
    def prop_to(self, v_prop):
        self.v_prop = v_prop

    # Step time
    def step_time(self, dt, v_prop, t_rd, wave, current):

        self.t += dt
        self.prop_to(v_prop)
        self.rudder_to(t_rd)
        self.set_grid()
        self.v_current_self = np.matmul(rotation_all_rev(-self.theta), current)

        # Linear forces and motion
        F_forward = self.F_prop() - self.D_forward() + self.W_forward(wave)
        F_lateral = -self.D_lateral() + self.W_lateral(wave)
        Fz = self.F_h(wave)
        Fz = 0
        F = np.array([F_forward, F_lateral, Fz])
        self.v_self =  + self.v_self + dt * F / self.M
        self.v = np.matmul(rotation_all(self.theta), self.v_self)
        self.pos += dt * self.v

        # Torque and angular motion
        T = np.array([self.T_r(wave, self.t), self.T_p(wave, self.t), self.T_y()])
        self.w = self.w + np.matmul(rotation_all(self.theta), dt * (T - self.D_w()) / self.I)
        self.theta = self.theta + dt * self.w

        return
