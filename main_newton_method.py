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
from optcon import NewtonMethod, GradientMethod
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
beta = 0.9
armijo_maxiters = 10 # number of Armijo iterations

visu_armijo = False

term_cond = 1e-6



#######################################
# Objects
#######################################

dyn = Dynamics()
ns, ni = dyn.ns, dyn.ni
QQt = np.eye(ns)*0.5e-3
QQt[1,1] = 1
# QQt = np.diag([1e-3,10,1e-3,1e-3,1e-3,1e-3])
RRt = 5e-4*np.eye(ni)
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
  slope = tt.shape[0]*1
  pp = p0+sigmoid_fcn(tt - tt[-1]/2,slope)[0]*(pT - p0)
  vv = sigmoid_fcn(tt - tt[-1]/2,slope)[1]*(pT - p0)

  return pp, vv


#######################################
# Reference curve
######################################
tt = np.linspace(0,tf,TT)

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

x0,z0,alpha0 = 0,0,6*np.pi/180
xf,zf,alphaf = 10,2,6*np.pi/180
vz = (zf-z0)/tf

# Get the two equillibrium point corresponding to some (theta, x, z) values
# dyn.get_equilibrium(Theta, X, Z)
# p1, in1 =  dyn.get_equilibrium(x0,z0,alpha0,tt)
# p2, in2 = dyn.get_equilibrium(xf,zf,alphaf,tt)
zz,zzd = reference_position(tt, z0, zf)
gg = np.ones(zz.shape)*20*np.pi/180


xx_ref[0,:] = x0+((xf-x0)/tf)*tt
# xx1_temp = np.roll((zzd/np.sin(gg))*dt,1)
# xx1_temp[0] = x0

# xx_ref[0,:] = np.cumsum(xx1_temp)
xx_ref[1,:] = zz.copy()
# xx_ref[2,:] = zzd/np.sin(gg)
# xx_ref[2,:] = 160/2
# xx_ref[3,:] = 1.2
# xx_ref[4,:] = 20
# xx_ref[5,:] = 0.5
plt.subplot(311)
plt.plot(tt,xx_ref[0,:])
plt.subplot(312)
plt.plot(tt,xx_ref[1,:])
plt.subplot(313)
plt.plot(tt,xx_ref[2,:])
plt.show()

cst = Cost(QQt,RRt,QQT)

GM = GradientMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond)
NM = NewtonMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond)

xx_init,uu_init = dyn.get_equilibrium(x0,z0,tt)
# xx_init = np.zeros((ns,TT))
# xx_init[2,:] = 50
# uu_init = np.zeros((ni,TT))
# print(xx_init.shape)
# for i in range(6):
#   plt.subplot(321+i)
#   plt.plot(xx_init[i,:])
# plt.show()

# plt.plot(uu_init[0,:])
# plt.show()

# xx_init, uu_init = GM.optimize(xx_init, uu_init, tf, dt)

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
