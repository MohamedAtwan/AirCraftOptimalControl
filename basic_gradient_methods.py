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
# max_iters = int(1200)
# stepsize = 1e-1
max_iters = int(2e3)
stepsize_0 = 1e-1

# ARMIJO PARAMETERS
cc = 0.5
# beta = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

term_cond = 1e-6

visu_armijo = False

term_cond = 1e-6

#######################################
# Trajectory parameters
#######################################

tf = 1 # final time in seconds
dt = 1e-3
dyn.dt = dt
# dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

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
  pp = p0+sigmoid_fcn(tt - tt[-1]/2,slope)[0]*(pT - p0)
  vv = sigmoid_fcn(tt - tt[-1]/2,slope)[1]*(pT - p0)

  return pp, vv

######################################
# Reference curve
######################################
tt = np.linspace(0,tf,TT)

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

# Get the two equillibrium point corresponding to some (theta, x, z) values
# dyn.get_equilibrium(Theta, X, Z)
p1, p2 =  dyn.get_equilibrium(20*np.pi/180,0,200), dyn.get_equilibrium(20*np.pi/180,10,210)
for i in range(6):
  xx_ref[i,:TT//2], xx_ref[i,TT//2:] = p1[i]*np.ones((TT//2)), p2[i]*np.ones((TT//2))

x0 = xx_ref[:,0]

######################################
# quasi static trajectory
######################################

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

# define some initial trajectory at k = 0 by forward integrating the initial value at equillibrium
xx[:,0,0] = x0.flatten()

for i in range(TT-1):
  xx[:,i+1,0] = dyn.step(xx[:,i,0],uu)[0].flatten()

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
    # print(BB)


    lmbd_temp = AA.T@lmbd[:,tt+1,kk][:,None] + aa       # costate equation
    deltau_temp = - BB.T@lmbd[:,tt+1,kk][:,None] - bb


    lmbd[:,tt,kk] = lmbd_temp.squeeze()
    deltau[:,tt,kk] = deltau_temp.squeeze()

    descent[kk] += (deltau[:,tt,kk].T@deltau[:,tt,kk])

  ##################################
  # Stepsize selection - ARMIJO
  ##################################


  stepsizes = []  # list of stepsizes
  costs_armijo = []

  stepsize = stepsize_0

  for ii in range(armijo_maxiters):

    # temp solution update

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
      xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

    # temp cost calculation
    JJ_temp = 0

    for tt in range(TT-1):
      temp_cost = dyn.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ_temp += temp_cost

    temp_cost = dyn.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost

    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize

    if JJ_temp > JJ[kk] - cc*stepsize*descent[kk]:
        # update the stepsize
        stepsize = beta*stepsize
    
    else:
        print('Armijo stepsize = {}'.format(stepsize))
        break


  ############################
  # Armijo plot
  ############################

  if visu_armijo:

    steps = np.linspace(0,1,int(1e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

      step = steps[ii]

      # temp solution update

      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = dyn.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ_temp += temp_cost

      temp_cost = dyn.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
      JJ_temp += temp_cost

      costs[ii] = JJ_temp


    plt.figure(1)
    plt.clf()

    plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ[kk] - cc*descent[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()

    plt.show()

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

  rr = max((max_iters-1-(max_iters-1)//2)/(max_iters-1),0)
  dyn.update_epsilon(rr)
  # print(rr)
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

labels = {0:'X', 1:'X_dot', 2: 'Z', 3: 'Z_dot', 4:'Theta', 5:'Theta_dot'}
for j in [0,2,4]:
  # fig, axs = plt.subplots(2, 1, sharex='all')
  plt.figure()
  for i in range(2):
    plt.subplot(211+i)
    plt.plot(tt_hor, xx_star[i+j,:], linewidth=2)
    plt.plot(tt_hor, xx_ref[i+j,:], 'g--', linewidth=2)
    plt.grid()
    plt.ylabel('{}'.format(labels[i+j]))
    # axs[i].plot(tt_hor, xx_star[i+j,:], linewidth=2)
    # axs[i].plot(tt_hor, xx_ref[i+j,:], 'g--', linewidth=2)
    # axs[i].grid()
    # axs[i].set_ylabel('$state_{}$'.format(labels[i+j]))


# fig, axs = plt.subplots(2, 1, sharex='all')

# for i in range(2):
#   axs[i].plot(tt_hor, xx_star[i+2,:], linewidth=2)
#   axs[i].plot(tt_hor, xx_ref[i+2,:], 'g--', linewidth=2)
#   axs[i].grid()
#   axs[i].set_ylabel('$x_{}$'.format(i))

# fig, axs = plt.subplots(2, 1, sharex='all')
plt.figure()
for i in range(2):
  plt.subplot(211+i)
  plt.plot(tt_hor, uu_star[i,:], linewidth=2)
  plt.plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
  plt.grid()
  plt.ylabel('U_{}'.format(i))
  # axs[i].plot(tt_hor, uu_star[i,:], linewidth=2)
  # axs[i].plot(tt_hor, uu_ref[i,:], 'g--', linewidth=2)
  # axs[i].grid()
  # axs[i].set_ylabel('$U_{}$'.format(i+1))

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
