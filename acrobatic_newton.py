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

import matplotlib.animation as animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) #minor grid
from animate import Airfoil



# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

# np.seterr('raise')

SAVE = True
visu_animation = True
#######################################
# Algorithm parameters
#######################################


max_iters = int(2e2)
stepsize_0 = 1
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
QQt = np.eye(ns)*5e-5
# QQt[1,1] = 3e-1
QQt[1,1] = 1e-1
# QQt = np.diag([1e-3,10,1e-3,1e-3,1e-3,1e-3])
RRt = 5e-5*np.eye(ni)
QQT = QQt.copy()
QQT[1,1] = QQt[1,1]*10#QQt.copy()*100
QQT[3,3] = QQT[1,1]
QQT[0,0] = QQT[1,1]

# QQT[1,1] = QQT[1,1]*100


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
  slope = tt.shape[0]*0.05
  TT = tt.shape[0]
  pp = np.zeros((TT,))
  vv = np.zeros((TT,))
  pp[:TT//2] = p0+sigmoid_fcn(tt[:TT//2] - tt[TT//2]/2,slope)[0]*(pT - p0)
  vv[:TT//2] = sigmoid_fcn(tt[:TT//2] - tt[TT//2]/2,slope)[1]*(pT - p0)
  pp[TT//2:] = p0+sigmoid_fcn(-tt[:TT//2] + tt[TT//2]/2,slope)[0]*(pT - p0)
  vv[TT//2:] = sigmoid_fcn(-tt[:TT//2] + tt[TT//2]/2,slope)[1]*(pT - p0)
  temp = pp.copy()
  N = pp.shape[0]
  pp = np.zeros(temp.shape)
  pp[int(0.05*N):int(0.50*N)] = temp[:int(0.45*N)].copy()
  pp[int(0.50*N):int(0.95*N)] = temp[-int(0.45*N):].copy()
  return pp, vv



#######################################
# Reference curve
######################################
tt = np.linspace(0,tf,TT)

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

x0,z0,alpha0 = 0,0,6*np.pi/180
xf,zf,alphaf = 8,3,6*np.pi/180
vz = (zf-z0)/tf



zz,zzd = reference_position(tt, z0, zf)
# xx,xxd = reference_position(tt, x0, xf)

xxe,uue = dyn.get_equilibrium(np.zeros(dyn.ns,),tt)
xx_ref[0,:] = x0+((xf-x0)/tf)*tt
xx_ref[1,:] = zz.copy()
for i in range(2,dyn.ns):
  xx_ref[i,:] = xxe[i]#(zzd**2+((xf-x0)/tf)**2)**0.5

xx_ref[3,:] = 0.0
for i in range(dyn.ni):
  uu_ref[i,:] = uue[i]
# for i in range(xx_ref.shape[1]):
#   xx_ref[5,i] = np.math.asin(-zzd[i]/xx_ref[2,i])
# xx0,uu0 = dyn.get_equilibrium_1(xx_ref[:,0],uu_ref[:,0])
# xx_ref[3,:] = xx0[3]
uu_ref[0,:] = uu_ref[0,:]*10
uu_ref[1,:] = -60


plt.subplot(221)
plt.plot(tt,xx_ref[0,:])
plt.subplot(222)
plt.plot(tt,xx_ref[1,:])
plt.subplot(223)
plt.plot(tt,xx_ref[2,:])
plt.subplot(224)
plt.plot(tt,xx_ref[3,:])
plt.show()

cst = Cost(QQt,RRt,QQT)

GM = GradientMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond)
NM = NewtonMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond)


# xx_init,uu_init = np.load('xx_star.npy'), np.load('uu_star.npy')
# xx_init = np.zeros((dyn.ns,TT))
# uu_init = np.zeros((dyn.ni,TT))
# uu_init[0,:] = 1000*np.sinc(10*(tt-tt[-1]/2))
# uu_init[1,:] = np.sinc(10*(tt-tt[-1]/2))
# xx_init[0,:] = np.linspace(x0,xf,TT)
# xx_init[1,:] = 3.71*np.sinc(10*(tt-tt[-1]/2))#np.zeros((TT,))
# xx_init[2,:] = np.ones((TT,))*xx_ref[2,0]
# xx_init[3,:] = np.ones((TT,))*xx_ref[3,0]
# xx_init[4,:] = np.zeros((TT,))
# xx_init[5,:] = np.ones((TT,))*xx_ref[5,0]

xx_init,uu_init = dyn.get_initial_trajectory(xx_ref,tt)
# uu_init[0,:] = uu_init[0,:]*100
# for i in range(6):
#   plt.subplot(321+i)
#   plt.plot(xx_init[i,:])
# plt.show()

# xx_init,uu_init = dyn.constant_input_trajectory(xx_init,tt)
for i in range(6):
  plt.subplot(321+i)
  plt.plot(xx_init[i,:])
plt.show()

for i in range(dyn.ni):
  plt.subplot(211+i)
  plt.plot(uu_init[i,:])
plt.show()



# xx_init, uu_init = GM.optimize(xx_init, uu_init, tf, dt)

xx_star, uu_star = NM.optimize(xx_init, uu_init, tf, dt)

if SAVE:
  np.save('Data/xx_star_acrobatic.npy',xx_star)
  np.save('Data/uu_star_acrobatic.npy',uu_star)




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

if visu_animation:
  limX = max(xf,zf)*1.1
  aircraft = Airfoil(1,xx_star,xx_ref,xlim = [0,15], ylim = [-5,5])
  aircraft.run_animation()

