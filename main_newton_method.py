import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import log
from random import random
from optcon import NewtonMethod, GradientMethod

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
armijo_maxiters = 10 # number of Armijo iterations

visu_armijo = False

term_cond = 1e-6



#######################################
# Objects & weights definition
#######################################

dyn = Dynamics()
ns, ni = dyn.ns, dyn.ni
QQt = np.eye(ns)*1e-6
QQt[1,1] = dyn.m*dyn.g*0.01
QQt[2,2] = 0.5*dyn.m*0.001
QQt[3,3] = 0.01
QQt[4,4] = 0.5*dyn.J*0.001

# QQt = np.diag([1e-3,10,1e-3,1e-3,1e-3,1e-3])
RRt = 1e-6*np.eye(ni)
QQT = QQt.copy()
QQT[1,1] = QQT[1,1]*20
QQT[3,3] = QQT[1,1]
QQT[0,0] = QQT[1,1]
print(QQt)


#######################################
# Trajectory parameters
#######################################

tf = 1 # final time in seconds
dt = 1e-3
dyn.dt = dt

TT = int(tf/dt) # discrete-time samples

#######################################
# Helper functions
#######################################
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
xf,zf,alphaf = 16,2.71,6*np.pi/180
vz = (zf-z0)/tf

zz,zzd = reference_position(tt, z0, zf)

xxe,uue = dyn.get_equilibrium(np.zeros(dyn.ns,),tt)
xx_ref[0,:] = x0+((xf-x0)/tf)*tt
xx_ref[1,:] = zz.copy()
xx_ref[2,:] = (zzd**2+((xf-x0)/tf)**2)**0.5


xx_ref[3,:] = 0.0
xx_ref[5,:] = 0.0

# plotting the reference curves
for i in range(dyn.ni):
  uu_ref[i,:] = uue[i]


plt.subplot(221)
plt.plot(tt,xx_ref[0,:])
plt.subplot(222)
plt.plot(tt,xx_ref[1,:])
plt.subplot(223)
plt.plot(tt,xx_ref[2,:])
plt.subplot(224)
plt.plot(tt,xx_ref[3,:])
plt.show()


#######################################
# Algorithm Objects definition
######################################
cst = Cost(QQt,RRt,QQT)

NM = NewtonMethod(dyn,cst,xx_ref,uu_ref, max_iters = max_iters,
                    stepsize_0 = stepsize_0, cc = cc, beta = beta,
                    armijo_maxiters = armijo_maxiters, term_cond = term_cond, visu_armijo = False)


#######################################
# calculation of the initial trajectory
# at the first iteration (k = 0)
######################################
xx_init,uu_init = dyn.get_initial_trajectory(xx_ref,tt)
for i in range(6):
  plt.subplot(321+i)
  plt.plot(xx_init[i,:])
plt.show()

#######################################
# Applying the algorithm
######################################
xx_star, uu_star = NM.optimize(xx_init, uu_init, tf, dt)

#######################################
# Saving the optimal trajectories
######################################
if SAVE:
  np.save('Data/xx_star.npy',xx_star)
  np.save('Data/uu_star.npy',uu_star)



#######################################
# Plotting
######################################
tt_hor = np.linspace(0,tf,TT)
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
  if j == 0:
    plt.savefig('Figures/X_Z_plot_step.png')
  elif j == 2:
    plt.savefig('Figures/V_theta_plot_step.png')
  elif j == 4:
    plt.savefig('Figures/q_gamma_plot_step.png')

plt.figure()
for i in range(2):
  plt.subplot(211+i)
  plt.plot(tt_hor, uu_star[i,:], linewidth=2)
  plt.plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
  plt.grid()
  plt.ylabel('U_{}'.format(i))
plt.show()


#######################################
# Animation
######################################
if visu_animation:
  limX = max(xf,zf)*1.1
  aircraft = Airfoil(20,xx_star,xx_ref,xlim = [0,17], ylim = [-5,5])
  aircraft.run_animation(name = 'step')
