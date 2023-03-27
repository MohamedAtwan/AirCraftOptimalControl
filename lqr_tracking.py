import numpy as np
import matplotlib.pyplot as plt
from aircraft_simplified import Dynamics, Cost

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
	TT = tt.shape[0]
	PP = np.zeros((*QQT.shape,TT))
	KK = np.zeros((ni,ns,TT-1))
	SS = np.zeros((ni,ns,1))
	uu_current = uu_opt[:,0]
	xx_current = xx_opt[:,0]
	xx_reg, uu_reg = np.zeros((ns,TT)), np.zeros((ni, TT))



	for t in range(TT-1):

		xxp, fx, fu = dyn.step(xx_current, uu_current)[0:3]

		AA = fx.T
		BB = fu.T
		delta_xx = xxp-xx_current

		Kt,_,delta_x, deltau = ltv_LQR(np.expand_dims(AA,-1),np.expand_dims(BB,-1),np.expand_dims(QQt,-1),
											  np.expand_dims(RRt,-1),SS,QQT,
											  1, delta_xx)
		uu_current = uu_opt[:,t] + deltau.flatten()
		xx_reg[:,t], uu_reg[:,t] = xx_current, uu_current
		xx_current = xxp
	xx_reg[:,t] = xx_current

	# lmbd, QQ_terminal = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1])[1:]
	# PP[:,:,TT-1] = QQT

	# for t in reversed(range(TT-1)):  # integration backward in time

	# 	# aa, bb, lxx, lxu, lux, luu = cst.stagecost(xx_opt[:,t], uu_opt[:,t], xx_ref[:,t], uu_ref[:,t])[1:]
	# 	fx, fu = dyn.step(xx_opt[:,t], uu_opt[:,t])[1:3]

	# 	AA = fx.T
	# 	BB = fu.T

	# 	# if kk > 8:
	# 	# 	QQt = lxx + fxx
	# 	# 	RRt = luu + fuu

	# 	# else:
	# 	# 	QQt = lxx
	# 	# 	RRt = luu
	# 	# 	# 28th march @ 17:30

	# 	# lmbd = AA.T@lmbd[:,None] + aa
	# 	PP[:,:,t] = QQt + AA.T@(PP[:,:,t+1]@AA) - (AA.T@PP[:,:,t+1]@BB)@(np.linalg.inv(RRt+BB.T@(PP[:,:,t+1]@BB))@(BB.T@(PP[:,:,t+1]@AA)))
		
	# for t in range(TT-1):
	# 	PPtp = PP[:,:,t+1]
	# 	# Check positive definiteness

	# 	MM = RRt + BB.T@PPtp@BB

	# 	if not np.all(np.linalg.eigvals(MM) > 0):
	# 	  MM += 0.5*np.eye(ni)

	# 	KK[:,:,t] = -np.linalg.inv(MM)@(BB.T@(PPtp@AA))

	# uu_reg = np.zeros(uu_opt.shape)
	# xx_reg = np.zeros(xx_opt.shape)
	# xt = xx_opt[:,0].copy()
	# xx_reg[:,0] = xt.copy()
	# for t in range(TT-1):
	# 	delta_x = xx_ref[:,t]-xt.flatten()
	# 	uu_reg[:,t] = ((uu_opt[:,t][:,None]-uu_ref[:,t][:,None])+KK[:,:,t]@delta_x[:,None]).flatten()
	# 	# xt = dyn.step(xt, uu_reg[:,t])[0]
	# 	fx, fu = dyn.step(xx_opt[:,t], uu_opt[:,t])[1:3]

	# 	AA = fx.T
	# 	BB = fu.T

	# 	xt = AA@xx-refhh-xt.flatten()[:,None]+BB@uu_reg[:,t][:,None]
	# 	xx_reg[:,t+1] = xt.flatten()
	return xx_reg, uu_reg


def plot_trajectory(xx_opt,uu_opt,xx_lqr,uu_lqr,tt):
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
	QQt = np.eye(ns)*100#*1e-5
	# QQt[1,1] = 1
	RRt = np.eye(ni)#*1e-5
	QQT = QQt.copy()
	# QQT[1,1] = QQt[1,1]*1#QQt.copy()*100
	# QQT[3,3] = QQT[1,1]
	# QQT[0,0] = QQT[1,1]
	# cst = Cost(QQt,RRt,QQT)
	xx_opt,uu_opt = np.load('xx_star.npy'), np.load('uu_star.npy')
	TT = xx_opt.shape[1]

	# #######################################
	# # Reference curve
	# ######################################
	tt = np.linspace(0,1,TT)
	tf = tt[-1]

	# xx_ref = np.zeros((ns, TT))
	# uu_ref = np.zeros((ni, TT))

	# x0,z0,alpha0 = 0,0,6*np.pi/180
	# xf,zf,alphaf = 16,2.71,6*np.pi/180
	# vz = (zf-z0)/tf



	# zz,zzd = reference_position(tt, z0, zf)

	# xxe,uue = dyn.get_equilibrium(np.zeros(dyn.ns,),tt)
	# xx_ref[0,:] = x0+((xf-x0)/tf)*tt
	# xx_ref[1,:] = zz.copy()
	# for i in range(2,dyn.ns):
	#   xx_ref[i,:] = xxe[i]

	# xx_ref[3,:] = 0.0
	# for i in range(dyn.ni):
	#   uu_ref[i,:] = uue[i]
	# # uu_ref[0,:] = uu_ref[0,:]*10
	# # uu_ref[1,:] = -60

	# plot_trajectory(xx_opt,uu_opt,xx_opt,uu_opt,tt)

	xx_lqr, uu_lqr = lqr_tracking(xx_opt, uu_opt, tt)
	plot_trajectory(xx_opt,uu_opt,xx_lqr,uu_lqr,tt)