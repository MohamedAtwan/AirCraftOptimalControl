#
# Gradient method for Optimal Control
# Main
# Lorenzo Sforni
# Bologna, 22/11/2022
#


import numpy as np
import matplotlib.pyplot as plt


from aircraft import Dynamics


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

dyn = Dynamics()

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

term_cond = 1e-6

visu_armijo = False

#######################################
# Trajectory parameters
#######################################

tf = 20 # final time in seconds

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt) # discrete-time samples


######################################
# Reference curve
######################################


ref_theta = 30
ref_angle_of_attack = 17


xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

xx_ref[1,:int(TT/2)] = np.ones((1,int(TT/2)))*180*np.cos(np.deg2rad(ref_theta-ref_angle_of_attack))
xx_ref[1,int(TT/2):] = np.ones((1,int(TT/2)))*180*np.cos(np.deg2rad(-ref_theta+ref_angle_of_attack))
xx_ref[3,:int(TT/2)] = -1* np.ones((1,int(TT/2)))*180*np.sin(np.deg2rad(ref_theta-ref_angle_of_attack))
xx_ref[3,int(TT/2):] = -1* np.ones((1,int(TT/2)))*180*np.sin(np.deg2rad(-ref_theta+ref_angle_of_attack))
xx_ref[4,:int(TT/2)] = np.ones((1,int(TT/2)))*np.deg2rad(ref_theta)
xx_ref[4,int(TT/2):] = np.ones((1,int(TT/2)))*np.deg2rad(-ref_theta)
xx_ref[6,:int(TT/2)] = np.ones((1,int(TT/2)))*np.deg2rad(ref_angle_of_attack)
xx_ref[6,int(TT/2):] = np.ones((1,int(TT/2)))*np.deg2rad(-ref_angle_of_attack)

for tt in range(TT):
  uu_ref[0,tt] = (-dyn.liftForce(xx_ref[:,tt])[0] + dyn.m*dyn.g*np.cos(xx_ref[6,tt]))/np.sin(xx_ref[4,tt]-xx_ref[6,tt])


x0 = xx_ref[:,0]

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

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

#     plt.pause(4)

    
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

fig, axs = plt.subplots(ns+ni, 1, sharex='all')


axs[0].plot(tt_hor, xx_star[0,:], linewidth=2)
axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
axs[0].grid()
axs[0].set_ylabel('$x_1$')

axs[1].plot(tt_hor, xx_star[1,:], linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$x_2$')

axs[2].plot(tt_hor, uu_star[0,:],'r', linewidth=2)
axs[2].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$u$')
axs[2].set_xlabel('time')
  

plt.show()
