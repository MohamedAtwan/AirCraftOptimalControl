#
# Gradient method for Optimal Control
# Main
# Lorenzo Sforni
# Bologna, 22/11/2022
#


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# import os

from aircraft_simplified import Dynamics


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

dyn = Dynamics()
# np.seterr('raise')

#######################################
# Algorithm parameters
#######################################

# max_iters = int(2e2)
max_iters = int(200)
stepsize = 1e-1

term_cond = 1e-6

#######################################
# Trajectory parameters
#######################################

tf = 0.1 # final time in seconds
dt = 1e-4
dyn.dt = dt
# dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt) # discrete-time samples

# NN = int(1e3) #number of points of the desired trajectory
######################################
# Define the desired velocity profile
######################################

def sigmoid_fcn(tt):
  """
    Sigmoid function

    Return
    - s = 1/1+e^-t
    - ds = d/dx s(t)
  """

  ss = 1/(1 + np.exp(0.02*(-tt)))

  ds = ss*(1-ss)

  return ss, ds


def reference_position(tt, p0, pT, T):
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

  pp = p0 + sigmoid_fcn(tt - T/2)[0]*(pT - p0)
  vv = p0 + sigmoid_fcn(tt - T/2)[1]*(pT - p0)

  return pp, vv

######################################
# Reference curve
######################################
tt = np.linspace(0,tf,TT)
p0 = 250 # ok for x and z
pT = 500 # ok for x and z

theta0 = np.deg2rad(48)
thetaT = np.deg2rad(40)

gamma0 = np.deg2rad(0)
gammaT = np.deg2rad(20)

px_ref, vx_ref = reference_position(tt, p0, pT, TT)
py_ref, vy_ref = reference_position(tt, p0, pT, TT)
theta_ref, thetad_ref = reference_position(tt, theta0, thetaT, TT)


xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

xx_ref[0] = px_ref
xx_ref[1] = vx_ref
xx_ref[2] = py_ref
xx_ref[3] = vy_ref
xx_ref[4] = theta_ref
xx_ref[5] = thetad_ref

# x0_in = np.zeros((ns, TT))
# u0 = np.zeros((ni, TT))
# u0[0, :] = np.ones((1, TT))*10
# u0[1, :] = np.ones((1, TT))*10
# x0_in[0, :] = np.ones((1, TT))*5
# x0_in[1, :] = np.ones((1, TT))*5
# x0_in[2, :] = np.ones((1, TT))*5
# x0_in[3, :] = np.ones((1, TT))*5
# # v = np.sqrt(x0[1, :]**2+x0[3, :]**2)
# x0_in[4, :] = np.ones((1, TT))*0
# x0_in[5, :] = np.ones((1, TT))*0

# x0_k = x0_in[:, :]
# x0 = x0_in[:, 0]

# for tt in range(TT):
#   uu_ref[0,tt] = (-dyn.liftForce(xx_ref[:,tt])[0] + dyn.m*dyn.g*np.cos(xx_ref[6,tt]))/np.sin(xx_ref[4,tt]-xx_ref[6,tt])


x0 = xx_ref[:,0]

######################################
# quasi static trajectory
######################################

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.
xx[1,:,0] = 120.0*np.ones((TT,))
xx[3,:,0] = 40.0*np.zeros((TT,))
xx[4,:,0] = 48.52*np.pi/180.0 *np.ones((TT,))

lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0
# xx[:,:, 0] = x0_k
for kk in range(max_iters-1):

  JJ[kk] = 0
  # calculate cost
  for tt in range(TT-1):
    temp_cost = dyn.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
    JJ[kk] += temp_cost
  
  temp_cost = dyn.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
  JJ[kk] += temp_cost


  ##################################
  # Descent direction calculation
  ##################################

  lmbd_temp = dyn.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

  for tt in reversed(range(TT-1)):  # integration backward in time

    aa, bb = dyn.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
    fx, fu = dyn.step(xx[:,tt,kk], uu[:,tt,kk])[1:]

    AA = fx.T
    BB = fu.T


    lmbd_temp = AA.T@lmbd[:,tt+1,kk][:,None] + aa       # costate equation
    deltau_temp = - BB.T@lmbd[:,tt+1,kk][:,None] - bb


    lmbd[:,tt,kk] = lmbd_temp.squeeze()
    deltau[:,tt,kk] = deltau_temp.squeeze()

    descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]

    
  ############################
  # Update the current solution
  ############################


  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))

  xx_temp[:,0] = x0

  for tt in range(TT-1):
    uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
    xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

  xx[:,:,kk+1] = xx_temp
  uu[:,:,kk+1] = uu_temp

  ############################
  # Termination condition
  ############################

  print('Iter = {}\t Descent = {}\t Cost = {}'.format(kk,descent[kk], JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk

    break

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plot

############################
# Plots
############################

# cost and descent

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


# optimal trajectory



tt_hor = np.linspace(0,tf,TT)

# plt.figure()

fig, axs = plt.subplots(4, 1, sharex='all')


for i in range(4):
  axs[i].plot(tt_hor, xx_star[i,:], linewidth=2)
  axs[i].plot(tt_hor, xx_ref[i,:], 'g--', linewidth=2)
  axs[i].grid()
  axs[i].set_ylabel('$x_{}$'.format(i))


fig, axs = plt.subplots(4, 1, sharex='all')

for i in range(4):
  axs[i].plot(tt_hor, xx_star[i+4,:], linewidth=2)
  axs[i].plot(tt_hor, xx_ref[i+4,:], 'g--', linewidth=2)
  axs[i].grid()
  axs[i].set_ylabel('$x_{}$'.format(i))

fig, axs = plt.subplots(2, 1, sharex='all')
for i in range(2):
  axs[i].plot(tt_hor, uu_star[i,:], linewidth=2)
  axs[i].plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
  axs[i].grid()
  axs[i].set_ylabel('$x_{}$'.format(i))

# axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
# axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
# axs[1].grid()
# axs[1].set_ylabel('$x_2$')

# axs[2].plot(tt_hor, xx_star[2,:], linewidth=2)
# axs[2].plot(tt_hor, xx_ref[2,:], 'g--', linewidth=2)
# axs[2].grid()
# axs[2].set_ylabel('$x_3$')

# axs[3].plot(tt_hor, xx_star[3,:], linewidth=2)
# axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
# axs[3].grid()
# axs[3].set_ylabel('$x_4$')

# axs[4].plot(tt_hor, xx_star[4,:], linewidth=2)
# axs[4].plot(tt_hor, xx_ref[4,:], 'g--', linewidth=2)
# axs[4].grid()
# axs[4].set_ylabel('$x_5$')

# axs[5].plot(tt_hor, xx_star[5,:], linewidth=2)
# axs[5].plot(tt_hor, xx_ref[5,:], 'g--', linewidth=2)
# axs[5].grid()
# axs[5].set_ylabel('$x_6$')

# axs[6].plot(tt_hor, xx_star[6,:], linewidth=2)
# axs[6].plot(tt_hor, xx_ref[6,:], 'g--', linewidth=2)
# axs[6].grid()
# axs[6].set_ylabel('$x_7$')

# axs[7].plot(tt_hor, xx_star[7,:], linewidth=2)
# axs[7].plot(tt_hor, xx_ref[7,:], 'g--', linewidth=2)
# axs[7].grid()
# axs[7].set_ylabel('$x_8$')

# axs[8].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
# axs[8].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
# axs[8].grid()
# axs[8].set_ylabel('$u0$')
# axs[8].set_xlabel('time')
  
# axs[9].plot(tt_hor, uu_star[1,:],'r', linewidth=2)
# axs[9].plot(tt_hor, uu_ref[1,:], 'r--', linewidth=2)
# axs[9].grid()
# axs[9].set_ylabel('$u1$')
# axs[9].set_xlabel('time')
  

plt.show()
