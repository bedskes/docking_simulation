import numpy as np

global rho
rho = 1005 # Density of sea water, kg/m^3
global g
g = 9.806 # Gravitational acceleration, m/s^2

class Wave:
    '''
    Wave objects model the heights and pressues from water waves in space and time.
    '''

    def __init__(self, A, k, omega, theta, phase):
        '''
        All of the inputs can either be floats or numpy arrays of equal length.

        A:      Amplitude, m
        k:      Wave number, rad/m
        omega:  Time frequency, rad/s
        theta:  Direction of propegation, measured from x-axis, rad
        phase:  Phase at t = 0, rad
        '''
        if len(np.shape(A)) == 0:
            self.kind = 'simple'
        else:
            self.kind = 'compound'
        self.A = A
        self.k = k
        self.omega = omega
        self.theta = theta
        self.phase = phase
    
    def height(self, x, y, t):
        if self.kind == 'simple':
            return wave_height(self.A, self.k, self.omega, self.theta, self.phase, x, y, t)
        else:
            return compound_wave_height(self.A, self.k, self.omega, self.theta, self.phase, x, y, t)
    
    def pressure(self, x, y, z, t):
        if self.kind == 'simple':
            return wave_pressure(self.A, self.k, self.omega, self.theta, self.phase, x, y, z, t)
        else:
            return compound_wave_pressure(self.A, self.k, self.omega, self.theta, self.phase, x, y, z, t)


def wave_height(A, k, omega, theta, phase, x, y, t):
    spacial_phase = k*(x*np.cos(theta) + y*np.sin(theta))
    n_space = np.shape(spacial_phase)
    time_phase = omega*t
    n_time = (np.product(np.shape([time_phase])),)
    spacial_phase = np.tile(spacial_phase, n_time + (1,)*len(n_space))
    time_phase = np.tile(time_phase, n_space + (1,)).T
    height = A*np.sin(spacial_phase + time_phase + phase)
    return np.squeeze(height)

def compound_wave_height(A, k, omega, theta, phase, x, y, t):
    height = 0
    for i in range(0, len(A)):
        height += wave_height(A[i], k[i], omega[i], theta[i], phase[i], x, y, t)
    return height

def wave_pressure(A, k, omega, theta, phase, x, y, z, t):
    spacial_phase = k*(x*np.cos(theta) + y*np.sin(theta))
    n_space = np.shape(spacial_phase)
    time_phase = omega*t
    n_time = (np.product(np.shape([time_phase])),)
    spacial_phase = np.tile(spacial_phase, n_time + (1,)*len(n_space))
    time_phase = np.tile(time_phase, n_space + (1,)).T
    pressure = A*rho*g*np.exp(k*z)*np.sin(spacial_phase + time_phase + phase)
    return np.squeeze(pressure)

def compound_wave_pressure(A, k, omega, theta, phase, x, y, z, t):
    pressure = 0
    for i in range(0, len(A)):
        pressure += wave_pressure(A[i], k[i], omega[i], theta[i], phase[i], x, y, z, t)
    return pressure