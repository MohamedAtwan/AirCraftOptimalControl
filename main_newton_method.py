#
# Gradient method for Optimal Control
# Main
# Lorenzo Sforni
# Bologna, 22/11/2022
#


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import log
from random import random
from optcon import NewtonMethod
# import os

from aircraft_simplified import Dynamics, Cost

import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

# np.seterr('raise')

#######################################
# Algorithm parameters
#######################################


max_iters = int(2e2)
stepsize_0 = 0.001

# ARMIJO PARAMETERS
cc = 0.5
# beta = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

visu_armijo = False

term_cond = 1e-6



#######################################
# Objects
#######################################

dyn = Dynamics()
ns, ni = dyn.ns, dyn.ni
QQt = np.eye(ns)
QQt[0,0] = 1
QQt[1,1] = 100
RRt = 1e-2*np.eye(ni)
QQT = 10*QQt


#######################################
# Trajectory parameters
#######################################

tf = 1 # final time in seconds
dt = 1e-3
dyn.dt = dt

TT = int(tf/dt) # discrete-time samples

# NN = int(1e3) #number of points of the desired trajectory
######################################
# Define the desired velocity profile
######################################

def sigmoid_fcn(tt,slope):
  """
    Sigmoid function

    Return
    - s = 1/1+e^-t
    - ds = d/dx s(t)
  """

  ss = 1/(1 + np.exp((-tt)*slope))

  ds = ss*(1-ss)

  return ss, ds


def reference_position(tt, p0, pT):
  """
  Returns the desired position and velocity for a smooth transition

  Args
    - tt time instant
    - p0 initial position
    - pT final position
    - T time horizon

  Returns
    - position p(t) = p0 + \sigma(t - T/2)*(pT - p0)
    - velocity v(t) = d/dt p(t) = \sigma'(t - T/2)*(pT - p0)
  """
  slope = tt.shape[0]*0.01
  pp = p0+sigmoid_fcn(tt - tt[-1]/2,slope)[0]*(pT - p0)
  vv = sigmoid_fcn(tt - tt[-1]/2,slope)[1]*(pT - p0)

  return pp, vv


#######################################
# Reference curve
######################################
tt = np.linspace(0,tf,TT)

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

x0,z0,alpha0 = 0,200,5*np.pi/180
xf,zf,alphaf = 20,220,5*np.pi/180
# Get the two equillibrium point corresponding to some (theta, x, z) values
# dyn.get_equilibrium(Theta, X, Z)
# p1, in1 =  dyn.get_equilibrium(x0,z0,alpha0,tt)
# p2, in2 = dyn.get_equilibrium(xf,zf,alphaf,tt)
zz,zzd = reference_position(tt, z0, zf)
xx_ref[0,:] = reference_position(tt, x0, xf)[0]
xx_ref[1,:] = zz

cst = Cost(QQt,RRt,QQT)
NM = NewtonMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond)

xx_init,uu_init = dyn.get_equilibrium(x0,z0,alpha0,tt)
xx_star, uu_star = NM.optimize(xx_init, uu_init, tf, dt)


tt_hor = np.linspace(0,tf,TT)

# plt.figure()

labels = {0:'X', 1:'Z', 2: 'V', 3: 'Theta', 4:'q', 5:'Gamma'}
for j in [0,2,4]:
  # fig, axs = plt.subplots(2, 1, sharex='all')
  plt.figure()
  for i in range(2):
    plt.subplot(211+i)
    plt.plot(tt_hor, xx_star[i+j,:], linewidth=2)
    plt.plot(tt_hor, xx_ref[i+j,:], 'g--', linewidth=2)
    plt.grid()
    plt.ylabel('{}'.format(labels[i+j]))

plt.figure()
for i in range(2):
  plt.subplot(211+i)
  plt.plot(tt_hor, uu_star[i,:], linewidth=2)
  plt.plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
  plt.grid()
  plt.ylabel('U_{}'.format(i))

  

plt.show()
