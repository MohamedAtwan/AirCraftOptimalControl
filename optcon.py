import numpy as np
import matplotlib.pyplot as plt


visu_armijo = False
class GradientMethod:
	def __init__(self,Dynamics,cost,xx_ref,uu_ref, max_iters = 200,
				     stepsize_0 = 1e-2, cc = 0.5, beta = 0.7,
				     armijo_maxiters = 20, term_cond = 1e-6,visu_armijo = False):

		self.dyn = Dynamics
		self.cst = cost
		self.ns, self.ni = self.dyn.ns, self.dyn.ni
		self.xx_ref, self.uu_ref = xx_ref, uu_ref
		self.max_iters = max_iters
		self.stepsize_0 = stepsize_0
		self.cc, self.beta = cc, beta
		self.term_cond = term_cond
		self.armijo_maxiters = armijo_maxiters
		self.visu_armijo = visu_armijo


	def optimize(self,xx_init, uu_init, tf, dt):
		ns, ni = self.ns, self.ni
		cst = self.cst
		dyn = self.dyn
		xx_ref, uu_ref = self.xx_ref, self.uu_ref
		TT = int(tf/dt)

		#######################################
		# Algorithm parameters
		#######################################


		max_iters = self.max_iters
		term_cond = 1e-6


		#######################################
		# Trajectory parameters
		#######################################

		# tf = 20 # final time in seconds

		# dt = dyn.dt   # get discretization step from dynamics
		ns = dyn.ns
		ni = dyn.ni

		TT = int(tf/dt) # discrete-time samples

		######################################
		# Arrays to store data
		######################################
		xx = np.zeros((ns, TT, max_iters))   # state seq.
		uu = np.zeros((ni, TT, max_iters))   # input seq.

		xx[:,:,0], uu[:,:,0] = xx_init, uu_init
		# plt.plot(xx[1,:,0])
		# plt.show()
		x0 = xx[:,0,0].copy()

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
				temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
				JJ[kk] += temp_cost

			temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
			JJ[kk] += temp_cost

			##################################
			# Descent direction calculation
			##################################
			lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1]
			lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

			for tt in reversed(range(TT-1)):  # integration backward in time

				aa, bb = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:3]
				fx, fu = dyn.step(xx[:,tt,kk], uu[:,tt,kk])[1:3]

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
			stepsize = self.armijo_stepsize(uu[:,:,kk],deltau[:,:,kk],xx_ref,uu_ref,x0,TT, JJ[kk], descent[kk])

			############################
			# Update the current solution
			############################

			xx_temp, uu_temp = self.get_update(stepsize,uu[:,:,kk],deltau[:,:,kk],x0)

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
		plt.show()
		return xx_star, uu_star

	def get_update(self,stepsize,uu,deltau,x0):
		xx_ref= self.xx_ref
		ns, ni = self.ns, self.ni
		dyn = self.dyn
		TT = uu.shape[1]

		xx_temp = np.zeros((ns,TT))
		uu_temp = np.zeros(uu.shape)
		xx_temp[:,0] = x0

		for tt in range(TT-1):
			uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
			xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

		return xx_temp, uu_temp


		
	def armijo_stepsize(self,uu,deltau,xx_ref,uu_ref,x0,TT, JJ, descent):
		ns, ni = self.ns, self.ni
		cst = self.cst
		dyn = self.dyn

		stepsize_0 = self.stepsize_0
		# descent = descent.squeeze()

		# ARMIJO PARAMETERS
		cc = self.cc
		# beta = 0.5
		beta = self.beta
		armijo_maxiters = self.armijo_maxiters # number of Armijo iterations


		##################################
		# Stepsize selection - ARMIJO
		##################################

		stepsizes = []  # list of stepsizes
		costs_armijo = []

		stepsize = stepsize_0
		BREAK = False

		for ii in range(armijo_maxiters):

			# temp solution update

			xx_temp = np.zeros((ns,TT))
			uu_temp = np.zeros((ni,TT))

			xx_temp[:,0] = x0

			for tt in range(TT-1):
				uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
				xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

			# temp cost calculation
			JJ_temp = 0

			for tt in range(TT-1):
				temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
				JJ_temp += temp_cost

			temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
			JJ_temp += temp_cost

			stepsizes.append(stepsize)      # save the stepsize
			costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize
			if JJ_temp > JJ - cc*stepsize*descent:
					# update the stepsize
					stepsize = beta*stepsize
			else:
				print('Armijo stepsize = {}'.format(stepsize))
				break
		# return stepsize

		# ############################
		# # Armijo plot
		# ############################

		if self.visu_armijo:

			steps = np.linspace(0,self.stepsize_0,int(self.armijo_maxiters))
			costs = np.zeros(len(steps))

			for ii in range(len(steps)):

				step = steps[ii]

				# temp solution update

				xx_temp = np.zeros((ns,TT))
				uu_temp = np.zeros((ni,TT))

				xx_temp[:,0] = x0

				for tt in range(TT-1):
					uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
					xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])[0]

				# temp cost calculation
				JJ_temp = 0

				for tt in range(TT-1):
					temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
					JJ_temp += temp_cost

				temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
				JJ_temp += temp_cost

				costs[ii] = JJ_temp
			plt.figure(1)
			plt.clf()

			plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
			plt.plot(steps, (JJ_temp - descent*steps).squeeze(), color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
			plt.plot(steps, (JJ_temp - cc*descent*steps).squeeze(), color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

			plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

			plt.grid()
			plt.xlabel('stepsize')
			plt.legend()
			plt.draw()

			plt.show()

		return stepsize

class NewtonMethod(GradientMethod):
	def __init__(self,Dynamics,cost,xx_ref,uu_ref, max_iters = 200,
				 stepsize_0 = 1e-2, cc = 0.5, beta = 0.7,
				 armijo_maxiters = 20, term_cond = 1e-6, visu_armijo = False):
		super(NewtonMethod,self).__init__(Dynamics,cost,xx_ref,uu_ref, max_iters,
										  stepsize_0, cc, beta,armijo_maxiters, term_cond,visu_armijo = visu_armijo)
		# self.dyn = Dynamics
		# self.cst = cost
		# self.ns, self.ni = self.dyn.ns, self.dyn.ni
		# self.xx_ref, self.uu_ref = xx_ref, uu_ref
		# self.max_iters = max_iters
		# self.stepsize_0 = stepsize_0
		# self.cc, self.beta = cc, beta
		# self.term_cond = term_cond
		# self.armijo_maxiters = armijo_maxiters

	def optimize(self,xx_init, uu_init, tf, dt):
		ns, ni = self.ns, self.ni
		cst = self.cst
		dyn = self.dyn
		xx_ref, uu_ref = self.xx_ref, self.uu_ref
		TT = int(tf/dt)

		#######################################
		# Algorithm parameters
		#######################################


		max_iters = self.max_iters
		term_cond = 1e-6


		#######################################
		# Trajectory parameters
		#######################################

		# tf = 20 # final time in seconds

		# dt = dyn.dt   # get discretization step from dynamics
		ns = dyn.ns
		ni = dyn.ni

		TT = int(tf/dt) # discrete-time samples

		######################################
		# Arrays to store data
		######################################
		xx = np.zeros((ns, TT, max_iters))   # state seq.
		uu = np.zeros((ni, TT, max_iters))   # input seq.
		QQ = np.zeros((ns,ns,TT,max_iters))
		RR = np.zeros((ni,ni,TT,max_iters))
		SS = np.zeros((ni,ns,TT,max_iters))
		qq = np.zeros((ns,TT,max_iters))
		rr = np.zeros((ni,TT,max_iters))
		AAin = np.zeros((ns,ns,TT,max_iters))
		BBin = np.zeros((ns,ni,TT,max_iters))



		xx[:,:,0], uu[:,:,0] = xx_init, uu_init
		# plt.plot(xx[1,:,0])
		# plt.show()
		x0 = xx[:,0,0].copy()

		lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.

		deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction

		JJ = np.zeros(max_iters)      # collect cost
		descent = np.zeros(max_iters) # collect descent direction

  		######################################
		# Main
		######################################
		print('-*-*-*-*-*-')

		kk = 0
		EE = np.eye(RR.shape[0])*10

		for kk in range(max_iters-1):
			# if kk > 60:
			# 	self.beta = 0.7

			JJ[kk] = 0
			# calculate cost
			for tt in range(TT-1):
				temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
				JJ[kk] += temp_cost

			temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
			JJ[kk] += temp_cost

			##################################
			# Descent direction calculation
			##################################
			lmbd_temp, QQ_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1:]
			lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
			QQ[:,:,TT-1,kk] = QQ_temp.squeeze()
			qq[:,TT-1,kk] = lmbd_temp.squeeze()

			for tt in reversed(range(TT-1)):  # integration backward in time

				aa, bb, lxx, lxu, lux, luu = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
				fx, fu, fxx, fuu, fux = dyn.step(xx[:,tt,kk], uu[:,tt,kk],lmbd[:,tt+1,kk])[1:]

				AA = fx.T
				BB = fu.T

				if kk > 8:
					# QQ[:,:,tt,kk] = lxx
					# RR[:,:,tt,kk] = luu
					# SS[:,:,tt,kk] = lux
					# RR[:,:,tt,kk] = RR[:,:,tt,kk]@ EE
					# self.stepsize_0 = 0.1
					QQ[:,:,tt,kk] = lxx + fxx
					RR[:,:,tt,kk] = luu + fuu
					SS[:,:,tt,kk] = lux + fux

				else:
					QQ[:,:,tt,kk] = lxx
					RR[:,:,tt,kk] = luu
					SS[:,:,tt,kk] = lux
					# 28th march @ 17:30 
				AAin[:,:,tt,kk] = AA.squeeze()
				BBin[:,:,tt,kk] = BB.squeeze()
				qq[:,tt,kk] = aa.squeeze()
				rr[:,tt,kk] = bb.squeeze()




				lmbd_temp = AA.T@lmbd[:,tt+1,kk][:,None] + aa       # costate equation


				lmbd[:,tt,kk] = lmbd_temp.squeeze()
				# deltau[:,tt,kk] = deltau_temp.squeeze()

				# descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]
			# print(lxx)
			# print(fxx)

			
			Kt,_,delta_x, deltau[:,:,kk] = ltv_LQR(AAin[:,:,:TT,kk],BBin[:,:,:TT,kk],QQ[:,:,:TT,kk],
											  RR[:,:,:TT,kk],SS[:,:,:TT,kk],QQ[:,:,TT-1,kk],
											  TT, np.zeros((self.ns)),qq[:,:TT,kk], rr[:,:TT,kk], qq[:,TT-1,kk])

			for tt in reversed(range(TT-1)):

				# descent[kk] += (deltau[:,tt,kk].T@deltau[:,tt,kk])
				tmp = BBin[:,:,tt,kk].T@lmbd[:,tt+1,kk][:,None] + rr[:,tt,kk][:,None]
				descent[kk] += tmp.T@tmp

			##################################
			# Stepsize selection - ARMIJO
			##################################
			stepsize = self.armijo_stepsize(uu[:,:,kk],deltau[:,:,kk],xx_ref,uu_ref,x0,TT, JJ[kk], descent[kk])
			# stepsize = self.stepsize_0

			############################
			# Update the current solution
			############################
			# for tt in range(TT-1):
			# 	uu[:,tt,kk+1] = uu[:,tt,kk]+stepsize*(Kt*delta_x)

			xx_temp, uu_temp = self.get_update(stepsize,uu[:,:,kk],deltau[:,:,kk],x0)

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
		return xx_star, uu_star



def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qq = None, rr = None, qqf = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  if qq is not None or rr is not None or qqf is not None:
    augmented = True
    print("Augmented term!")

  if augmented:
    if qq is None:
      qq = np.zeros(ns)

    if rr is None:
      rr = np.zeros(ni)

    if qqf is None:
      qqf = np.zeros(ns)

    # Check sizes

    try:  
      na, la = qq.shape
    except:
      qq = qq[:,None]
      na, la = qq.shape

    try:  
      nb, lb = rr.shape
    except:
      rr = rr[:,None]
      nb, lb = rr.shape

    if na != ns:
        print("State affine term does not match states dimension")
        exit()
    if nb != ni:
        print("Input affine term does not match inputs dimension")
        exit()
    if la == 1:
        qq = qq.repeat(TT, axis=1)
    if lb == 1:
        rr = rr.repeat(TT, axis=1)

  # Build matrices

  if augmented:

    KK = np.zeros((ni, ns + 1, TT))
    PP = np.zeros((ns+1, ns+1, TT))

    QQ = np.zeros((ns + 1, ns + 1, TT))
    QQf = np.zeros((ns + 1, ns + 1))
    SS = np.zeros((ni, ns + 1, TT))
    RR = np.zeros((ni, ni, TT))  # Must be positive definite

    AA = np.zeros((ns + 1, ns + 1, TT))
    BB = np.zeros((ns + 1, ni, TT))

    # Augmented matrices
    for tt in range(TT):

      # Cost

      QQ[1:, 0, tt] = 0.5 * qq[:,tt]
      QQ[0, 1:, tt] = 0.5 * qq[:,tt].T
      QQ[1:, 1:, tt] = QQin[:, :, tt]

      RR[:, :, tt] = RRin[:, :, tt]

      SS[:, 0, tt] = 0.5 * rr[:, tt]
      SS[:,1:,tt] = SSin[:, :, tt]

      # System

      AA[0, 0, tt] = 1
      AA[1:, 1:, tt] = AAin[:, :, tt]
      BB[1:, :, tt] = BBin[:, :, tt]

    QQf[1:, 0] = 0.5 * qqf
    QQf[0, 1:] = 0.5 * qqf.T
    QQf[1:, 1:] = QQfin

    # Systems trajectory
    xx = np.zeros((ns + 1, TT))
    uu = np.zeros((ni, TT))
    # Augmented state
    xx[0, :].fill(1)
    xx[1:,0] = x0

  else:
    KK = np.zeros((ni, ns, TT))
    PP = np.zeros((ns, ns, TT))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (BBt.T@PPtp@AAt + SSt).T@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt + SSt)
  
  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]

    # Check positive definiteness

    MM = RRt + BBt.T@PPtp@BBt

    if not np.all(np.linalg.eigvals(MM) > 0):

      # Regularization
      # print('regularization at tt = {}'.format(tt))
      MM += 0.5*np.eye(ni)

    KK[:,:,tt] = -np.linalg.inv(MM)@(BBt.T@PPtp@AAt + SSt)


  

  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt]@xx[:, tt]
    xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

    xx[:,tt+1] = xx_p.squeeze()

    if augmented:
      xxout = xx[1:,:]
    else:
      xxout = xx

    uuout = uu

  return KK, PP, xxout, uuout