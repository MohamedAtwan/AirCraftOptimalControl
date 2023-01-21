import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve



def constraints(x):
    # T =  x[1]
    # v = x[0]
    T =  160
    v = 30
    m = 12
    rho = 1.2
    s=0.61
    cd0 = 0.1716
    cda = 2.395
    cla = 3.256
    g=9.81
    J=0.24

    # gamma = np.deg2rad(10)
    # alfa = np.deg2rad(7)
    gamma = x[0]
    alfa = x[1]
    # print("gamma", gamma)
    # print("alfa", alfa)

    #x[0] = v
    #x[1] = T
    return [ 
            (-rho*v**2*s*(cd0+cda*alfa**2))/m - g*np.sin(gamma) + T*np.cos(alfa)/m,
            rho*v*s*cla*alfa/(2*m) - g * np.cos(gamma)/v + T*np.sin(alfa)/(m*v)
             ]
             
######################################
# Main
######################################
random_floats = np.random.rand(2)
root = fsolve(constraints, random_floats)

gamma = np.rad2deg(root[0])
alfa = np.rad2deg(root[1])
print("gamma {} alfa {}".format(gamma, alfa))
