import numpy as np
import matplotlib.pyplot as plt
from aircraft_simplified import Dynamics, Cost


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


def lqr_tracking(xx_opt, uu_opt, tt):
	'''
		Apply LQR tracing algorithm on the passed pair of states and outputs trajectory.
		The dynamics is linearized about each point of the trajectory and LQR input is used to track the trajectory

		Input: xx_opt --> optimal trajectory for the states
		Outputs: uu_opt --> optimal trajectory for the outputs
	'''
	TT = tt.shape[0]
	PP = np.zeros((*QQT.shape,TT))
	SS = np.zeros((ni,ns,TT))
	xx_reg, uu_reg = np.zeros((ns,TT)), np.zeros((ni, TT))

	# perturbation
	delta_xx = np.ones((6,))*0.1

	# Initilaztion
	AA=np.zeros([6,6,TT])
	BB=np.zeros([6,2,TT])
	KK = np.zeros((ni, ns, TT))
	xx_reg[:,0] = xx_opt[:,0] + delta_xx[:]

	# Iteration for each time step
	for t in range(TT):
		xxp, fx, fu = dyn.step(xx_opt[:,t], uu_opt[:,t])[0:3] # Simulation at time t

		# The Jaccobian
		AA[:,:,t] = fx.T
		BB[:,:,t] = fu.T
	
	# Calculation of the LQR state-feedback gain at each time step
	KK[:,:,:] = ltv_LQR(AA,BB,QQt,RRt,SS,QQT, TT, delta_xx,None, None, None)[0]

	# Applying the tracking at every time step
	for tt in range(TT-1):
		uu_reg[:, tt] = uu_opt[:,tt] + KK[:,:,tt]@(xx_reg[:, tt] - xx_opt[:,tt]) # input update
		xx_reg[:,tt+1] = dyn.step(xx_reg[:,tt], uu_reg[:,tt])[0] # Simulation at time t

	return xx_reg, uu_reg


def plot_trajectory(xx_opt,uu_opt,xx_lqr,uu_lqr,tt):
	'''
		Plots both the optimal and tracking trajectories

		Inputs: xx_opt --> the optimal state trajectory
				uu_opt --> the optimal input trajectory
				xx_lqr --> the Actual state trajectory after applying the LQR tracking
				uu_lqr --> the Actual input trajectory after applying the LQR tracking
				tt --> time vector
	'''

	# plotting the trajectory
	plt.figure()
	for i in range(ns):
		plt.subplot(321+i)
		plt.plot(tt,xx_opt[i,:],'g--',linewidth = 2, label = 'xx_opt')
		plt.plot(tt,xx_lqr[i,:],linewidth = 2, label = 'xx_reg')
		plt.legend()
		plt.grid()
		plt.title('State Trajectory')

	plt.figure()
	for i in range(ni):
		plt.subplot(211+i)
		plt.plot(tt[:-1],uu_opt[i,:-1],'g--',linewidth = 2, label = 'uu_opt')
		plt.plot(tt[:-1],uu_lqr[i,:-1],linewidth = 2, label = 'uu_reg')
		plt.legend()
		plt.grid()
		plt.title('Input Trajectory')

	plt.show()


if __name__ == '__main__':
	dyn = Dynamics()
	ns, ni = dyn.ns, dyn.ni
	QQt = np.eye(ns)*1e-5
	QQt[1,1] = 0.01
	RRt = np.eye(ni)*1e-5
	QQT = QQt.copy()
	xx_opt,uu_opt = np.load('xx_star.npy'), np.load('uu_star.npy')
	TT = xx_opt.shape[1]

	# #######################################
	# # solution for the lqr problem
	# ######################################
	tt = np.linspace(0,1,TT)
	tf = tt[-1]
	xx_lqr, uu_lqr = lqr_tracking(xx_opt, uu_opt, tt)

	# #######################################
	# # lqr plot wrt the opt trajectory
	# ######################################
	plot_trajectory(xx_opt,uu_opt,xx_lqr,uu_lqr,tt)