import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, least_squares
from aircraft import Dynamics



def dynamic_constraints(x):
    # T =  x[1]
    # v = x[0]

    # gamma = np.deg2rad(10)
    # alfa = np.deg2rad(7)
    gamma = x[0]
    alfa = Theta - gamma
    v = x[1]
    # print("gamma", gamma)
    # print("alfa", alfa)

    #x[0] = v
    #x[1] = T
    return [ 
            (-rho*v**2*s*(cd0+cda*alfa**2))/m - g*np.sin(gamma) + T*np.cos(alfa)/m,
            rho*v*s*cla*alfa/(2*m) - g * np.cos(gamma)/v + T*np.sin(alfa)/(m*v)
             ]

def print_states(Theta, gamma, alpha, V, x_val, z_val):
    xx = np.zeros((8,))
    xx[0] = x_init
    xx[1] = 
             
######################################
# Main
######################################
if __name__ == '__main__':
    DEG2RAD = np.pi/180
    dyn = Dynamics()
    T =  160
    Theta_init = 25*DEG2RAD
    Theta_final = 20*DEG2RAD
    x_init, z_init = 5000
    m = dyn.m
    rho = dyn.rho
    s = dyn.S
    cd0 = dyn.Cd0
    cda = dyn.Cda
    cla = dyn.Cla
    g = dyn.g
    J = dyn.J

    for Theta in [Theta_init, Theta_final]:
        random_floats = np.random.rand(2)
        root = fsolve(constraints, random_floats)

        gamma = np.rad2deg(root[0])
        alfa = np.rad2deg(root[1])
        print("gamma {} alfa {}".format(gamma, alfa))
