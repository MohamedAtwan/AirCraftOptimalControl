import numpy as np
from math import log
from scipy.optimize import fsolve, least_squares, minimize, NonlinearConstraint
from random import random

def round_theta(th):
	''' Some recursive algorithm to round theta to be between zero and 2pi'''
	if abs(th) <= 2*np.pi:
		return th
	else:
		if th < -2*np.pi:
			return round_theta(th + np.pi*2)
		else:
			return round_theta(th - np.pi*2)

class Cost:
	'''
		This class contains both the calls for the stage and the terminal cost.
	'''
	def __init__(self,QQt, RRt, QQT):
		self.QQt = QQt
		self.RRt = RRt
		self.QQT = QQT

	def stagecost(self,xx,uu, xx_ref, uu_ref):

		"""
		Stage-cost 

		Quadratic cost function 
		l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

		Args
		  - xx \in \R^2 state at time t
		  - xx_ref \in \R^2 state reference at time t

		  - uu \in \R^1 input at time t
		  - uu_ref \in \R^2 input reference at time t


		Return 
		  - cost at xx,uu
		  - gradient of l wrt x, at xx,uu
		  - gradient of l wrt u, at xx,uu

		"""

		QQt = self.QQt
		RRt = self.RRt
		xx = xx.reshape(-1,1)
		uu = uu.reshape(-1,1)

		xx_ref = xx_ref.reshape(-1,1)
		uu_ref = uu_ref.reshape(-1,1)

		ns = QQt.shape[0]
		ni = RRt.shape[0]

		# bar, dbar = self.barrier_function(xx,uu)

		ll = 0.5*(xx - xx_ref).T@(QQt@(xx - xx_ref)) + 0.5*(uu - uu_ref).T@(RRt@(uu - uu_ref))# + bar

		lx = (QQt@(xx - xx_ref))
		lu = RRt@(uu - uu_ref)
		lxx = QQt.copy()
		lxu = np.zeros((ns,ni))
		lux = np.zeros((ni,ns))
		luu = RRt.copy()
		return ll, lx, lu, lxx, lxu,lux, luu

	def termcost(self,xx,xx_ref):
		"""
		Terminal-cost

		Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

		Args
		- xx \in \R^2 state at time t
		- xx_ref \in \R^2 state reference at time t

		Return 
		- cost at xx,uu
		- gradient of l wrt x, at xx,uu
		- gradient of l wrt u, at xx,uu

		"""
		QQT = self.QQT

		xx = xx[:,None]
		xx_ref = xx_ref[:,None]

		llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

		lTx = QQT@(xx - xx_ref)
		lTxx = QQT

		return llT, lTx, lTxx



class Dynamics:
	def __init__(self):
		
		'''
			This class defines the dynamics for the planar aircraft model
		'''
		# The paramaters for the aerodynamics and the aircraft
		self.cd0 = 0.1716
		self.cda = 2.395
		self.cla = 3.256
		self.m = 12
		self.g = 9.81
		self.S = 0.61
		self.rho = 1.2
		self.J = 0.24
		self.ns = 6
		self.ni = 2
		self.dt = 1e-3
		
		self.Temp = None
		self.eps_init = 1.5
		self.eps_end = 0.1
		self.speedLimit = 480
		self.epsilon = self.eps_init

	def get_initial_trajectory(self,xx_ref,tt):
		'''
			This function calculates the initial trajectory starting from an equillibrium point
			Note: It is assumed the initial point of the reference trajectory to be the same of the
				  simulated dynamics
			Inputs: xx_ref --> The reference trajectory, tt --> the time vector
			Outputs: xx --> The trajectory for the states,  uu --> The trajectory for the inputs
		'''
		kp = 5
		kt = 2.5
		TT = tt.shape[0]
		uu = np.zeros((self.ni,TT))
		xx = np.zeros((self.ns,TT))
		x_temp = xx_ref[:,0].copy()
		xx[:,0] = x_temp.copy()
		uu_temp = np.zeros((self.ni,))
		for i in range(TT-1):
			uu_temp[0] = kp*((x_temp[0]-xx_ref[0,i+1])+(x_temp[1]-xx_ref[1,i+1]))
			uu_temp[1] = kt*((x_temp[3]-xx_ref[3,i+1])+((x_temp[5]-xx_ref[5,i+1])))
			x_temp = self.step(x_temp,uu_temp)[0]
			xx[:,i+1] = x_temp.copy()
			uu[:,i] = uu_temp.copy()
		return xx,uu



	def get_equilibrium(self,x0,tt):
		'''
			This calculates the equillibrium point
			Inputs: x0 --> initial guess for the equillibrium point, tt --> time vector
			outputs: xx --> the states of the calculated equillibrium point, uu --> the inputs of the calculated equillibrium point
		'''
		m = self.m
		J = self.J
		rho = self.rho
		Cla = self.cla
		S = self.S
		Cd0 = self.cd0
		Cda = self.cda
		g = self.g
		TT = tt.shape[0]
		dt = self.dt
		ns = self.ns 
		ni = self.ni
		x_init = np.array([10,0,0,0])
		xx, uu = x0.copy(), x_init
		self.Temp = xx.copy()
		sol = least_squares(self.__equilibrium_cost,x_init,bounds = [(-50,0,-np.pi,-np.pi),(50,1000,np.pi,np.pi)])
		uu[0] = sol.x[1]
		xx[2] = sol.x[0]
		xx[3] = sol.x[2]
		xx[5] = sol.x[3]
		return xx,uu
			
	def __equilibrium_cost(self,x):
		'''
			Calculates the cost of the optimization parameters for penalizing the equillibrium condition
			using a least squares optimization algorithm
		'''
		m = self.m
		g = self.g
		rho = self.rho 
		Cla = self.cla 
		Cda = self.cda 
		Cd0 = self.cd0
		S = self.S
		tf = 1
		NN = int(tf/self.dt)
		xx = np.zeros((self.ns,NN))
		xx = self.Temp.copy()
		xx[2] = x[0]
		T = x[1]
		xx[3] = x[2]
		xx[5] = x[3]
		alpha = xx[3]-xx[5]
		D,_ = self.dragForce(xx)
		L,_ = self.liftForce(xx)


		cost = [-D - m*g*np.sin(xx[5]) + T * np.cos(alpha),
				(L-m*g*np.cos(xx[5])+T*np.sin(alpha))]
		return cost




	def dragForce(self, xx):
		'''
			This function calculates the drag force value and the gradients of this force w.r.t. the system states
			Inputs: xx --> system states
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
		'''
		xx = xx.reshape(-1,1)
		# parameters
		ns = self.ns
		rho = self.rho
		S = self.S
		Cd0 = self.cd0
		Cda = self.cda

		alpha = xx[3,0] - xx[5,0]
		# Drag force value
		D = 0.5 * rho * (xx[2,0]**2) * S *(Cd0 + Cda * alpha**2)

		# Gradients
		dD_x = np.zeros((self.ns,1))
		dD_x[2,0] = rho * xx[2,0] * S *(Cd0 + Cda * alpha**2)
		dD_x[3,0] = rho * (xx[2,0]**2) * S * Cda * alpha
		dD_x[5,0] = - rho * (xx[2,0]**2) * S * Cda * alpha

		return D,dD_x

	def liftForce(self,xx):
		'''
			This function calculates the lift force value and the gradients of this force w.r.t. the system states
			Inputs: xx --> system states
			Outputs: D --> Lift force value,  dD_x --> gradients of the lift force w.r.t the system states
		'''
		xx = xx.reshape(-1,1)
		# parameters
		ns = self.ns
		rho = self.rho
		Cla = self.cla
		S = self.S
		alpha = xx[3,0] - xx[5,0]

		# Lift Force value
		L = 0.5*rho*(xx[2,0]**2)*S*Cla*alpha

		# Gradients
		dL_x = np.zeros((self.ns,1))
		dL_x[2,0] = rho * xx[2,0] * S * Cla * alpha
		dL_x[3,0] = 0.5 * rho * (xx[2,0]**2) * S * Cla
		dL_x[5,0] = - 0.5 * rho * (xx[2,0]**2) * S * Cla

		return L, dL_x

	def step(self,xx,uu,*args):
		# X, Z, V, THETA, Q, GAMMA
		# print(uu)
		# sign = -1 if xx[1] < 0 else 1
		# xx[1] = np.min((abs(xx[1]),480))*sign
		# sign = -1 if xx[3] < 0 else 1
		# xx[3] = np.min((abs(xx[3]),480))*sign
		'''
			This function takes on step discrete time step based on the current state and the current input
			Inputs: xx --> system states,  uu--> system Inputs, an extra optional input can be passed which is the costate variable lmbd
					lmbd --> costate variable
			Outputs: xxp, fx, fu, fxx, fuu, fux
		'''
		ns, ni = self.ns, self.ni
		xx = xx.reshape(-1,1)
		uu = uu.reshape(-1,1)
		if args:
			lmbd = args[0]
			if lmbd.shape[0] < ns:
				lmbd = np.atleast_2d(lmbd).T


		# parameters
		m = self.m
		J = self.J
		rho = self.rho
		Cla = self.cla
		S = self.S
		Cd0 = self.cd0
		Cda = self.cda
		g = self.g
		dt = self.dt
		alpha = xx[3,0] - xx[5,0]
		D, dD_x = self.dragForce(xx)
		L,dL_x = self.liftForce(xx)

		# contianer
		xxp = np.zeros((ns,),dtype = np.float32)

		# Discritized dynamics
		xxp[0] = xx[0,0] + dt*xx[2,0]*np.cos(xx[5,0])
		xxp[1] = xx[1,0] - dt*xx[2,0]*np.sin(xx[5,0])

		xxp[2] = xx[2,0] + (dt/m) * (-D - m*g*np.sin(xx[5,0]) + uu[0,0] * np.cos(alpha))
		xxp[3] = xx[3,0] + dt*xx[4,0]

		xxp[4] = xx[4,0] + dt * (uu[1,0]/J)
		xxp[5] = xx[5,0] + (dt/(m*xx[2,0]))*(L-m*g*np.cos(xx[5,0])+uu[0,0]*np.sin(alpha))

		# Gradients
		fx = np.zeros((ns, ns))
		fu = np.zeros((ni, ns))

		fx = np.array([[1, 0, dt*np.cos(xx[5,0]), 0, 0, -dt*xx[2,0]*np.sin(xx[5,0])],
					   [0, 1, -dt*np.sin(xx[5,0]), 0, 0, -dt*xx[2,0]*np.cos(xx[5,0])],
					   [0, 0, 1 - (S*dt*rho*xx[2,0]*(Cd0 + Cda*(xx[3,0] - xx[5,0])**2))/m, -(dt*((Cda*S*rho*(2*xx[3,0] - 2*xx[5,0])*xx[2,0]**2)/2 + uu[0,0]*np.sin(xx[3,0] - xx[5,0])))/m, 0, (dt*((Cda*S*rho*(2*xx[3,0] - 2*xx[5,0])*xx[2,0]**2)/2 + uu[0,0]*np.sin(xx[3,0] - xx[5,0]) - g*m*np.cos(xx[5,0])))/m],
					   [0, 0, 0, 1, dt, 0],
					   [0, 0, 0, 0, 1, 0],
					   [0, 0, (Cla*S*dt*rho*(xx[3,0] - xx[5,0]))/m - (dt*((Cla*S*rho*(xx[3,0] - xx[5,0])*xx[2,0]**2)/2 + uu[0,0]*np.sin(xx[3,0] - xx[5,0]) - g*m*np.cos(xx[5,0])))/(m*xx[2,0]**2), (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/(m*xx[2,0]), 0, 1 - (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0]) - g*m*np.sin(xx[5,0])))/(m*xx[2,0])]])
		fx = fx.T

		fu = np.array([[0, 0], [0, 0], [(dt*np.cos(xx[3,0] - xx[5,0]))/m, 0], [0, 0], [0, dt/J], [(dt*np.sin(xx[3,0] - xx[5,0]))/(m*xx[2,0]), 0]])
		fu = fu.T
		



		# Hessian --> napla_xx
		fxx = np.zeros((self.ns,self.ns,self.ns))
		fxx1 = np.zeros((self.ns,self.ns))
		fxx2 = np.zeros((self.ns,self.ns))
		fxx3 = np.zeros((self.ns,self.ns))
		fxx4 = np.zeros((self.ns,self.ns))
		fxx5 = np.zeros((self.ns,self.ns))
		fxx6 = np.zeros((self.ns,self.ns))

		fxx1 = np.array([[0, 0,0, 0, 0,0],
						 [0, 0,0, 0, 0,0],
						 [0, 0,0, 0, 0,-dt*np.sin(xx[5,0])],
						 [0, 0,0, 0, 0,0],
						 [0, 0,0, 0, 0,0],
						 [0, 0, -dt*np.sin(xx[5,0]), 0, 0, -dt*xx[2,0]*np.cos(xx[5,0])]])


		fxx2 = np.array([[0, 0, 0, 0, 0,0],
						 [0, 0, 0, 0, 0,0],
						 [0, 0, 0, 0, 0,-dt*np.cos(xx[5,0])],
						 [0, 0, 0, 0, 0,0],
						 [0, 0, 0, 0, 0,0],
						 [0, 0, -dt*np.cos(xx[5,0]), 0, 0, dt*xx[2,0]*np.sin(xx[5,0])]])

		fxx3 = np.array([[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0,-(S*dt*rho*(Cd0 + Cda*(xx[3,0] - xx[5,0])**2))/m,-(Cda*S*dt*rho*xx[2,0]*(2*xx[3,0] - 2*xx[5,0]))/m, 0,(Cda*S*dt*rho*xx[2,0]*(2*xx[3,0] - 2*xx[5,0]))/m],
						[0, 0,-(Cda*S*dt*rho*xx[2,0]*(2*xx[3,0] - 2*xx[5,0]))/m, -(dt*(Cda*S*rho*xx[2,0]**2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/m, 0,(dt*(Cda*S*rho*xx[2,0]**2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/m],
						[0, 0, 0, 0, 0, 0],
						[0, 0,(Cda*S*dt*rho*xx[2,0]*(2*xx[3,0] - 2*xx[5,0]))/m,  (dt*(Cda*S*rho*xx[2,0]**2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/m, 0, -(dt*(Cda*S*rho*xx[2,0]**2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0]) - g*m*np.sin(xx[5,0])))/m]])

		fxx6 = np.array([[0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0],
						[0, 0, (2*dt*((Cla*S*rho*(xx[3,0] - xx[5,0])*xx[2,0]**2)/2 + uu[0,0]*np.sin(xx[3,0] - xx[5,0]) - g*m*np.cos(xx[5,0])))/(m*xx[2,0]**3) - (Cla*S*dt*rho*(xx[3,0] - xx[5,0]))/(m*xx[2,0]), (Cla*S*dt*rho)/m - (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/(m*xx[2,0]**2), 0, (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0]) - g*m*np.sin(xx[5,0])))/(m*xx[2,0]**2) - (Cla*S*dt*rho)/m],
						[0, 0,(Cla*S*dt*rho)/m - (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0])))/(m*xx[2,0]**2), -(dt*uu[0,0]*np.sin(xx[3,0] - xx[5,0]))/(m*xx[2,0]), 0, (dt*uu[0,0]*np.sin(xx[3,0] - xx[5,0]))/(m*xx[2,0])],
						[0, 0, 0, 0, 0, 0],
						[0, 0, (dt*((Cla*S*rho*xx[2,0]**2)/2 + uu[0,0]*np.cos(xx[3,0] - xx[5,0]) - g*m*np.sin(xx[5,0])))/(m*xx[2,0]**2) - (Cla*S*dt*rho)/m, (dt*uu[0,0]*np.sin(xx[3,0] - xx[5,0]))/(m*xx[2,0]), 0, -(dt*(uu[0,0]*np.sin(xx[3,0] - xx[5,0]) - g*m*np.cos(xx[5,0])))/(m*xx[2,0])]])

		fxx[:,:,0] = fxx1
		fxx[:,:,1] = fxx2
		fxx[:,:,2] = fxx3
		fxx[:,:,5] = fxx6


		# Hessian --> napla_ux
		fux = np.zeros((self.ni,self.ns,self.ns))
		fux[:,:,2] = np.array([[0, 0, 0, -(dt*np.sin(xx[3,0] - xx[5,0]))/m, 0, (dt*np.sin(xx[3,0] - xx[5,0]))/m],
							   [0, 0, 0, 0, 0, 0]])
		fux[:,:,5] = np.array([[0, 0, -(dt*np.sin(xx[3,0] - xx[5,0]))/(m*xx[2,0]**2), (dt*np.cos(xx[3,0] - xx[5,0]))/(m*xx[2,0]), 0, -(dt*np.cos(xx[3,0] - xx[5,0]))/(m*xx[2,0])],
							  [0, 0, 0, 0, 0, 0]])

		# Hessian --> napla_uu
		fuu = np.zeros((self.ni,self.ni,self.ns))

		if args:
			# Tensor contraction with the costate variable lmbd
			fxx = tensorCont(fxx,lmbd)
			fuu = tensorCont(fuu,lmbd)
			fux = tensorCont(fux,lmbd)

			# fxx = (fxx@lmbd).squeeze()
			# fuu = (fuu@lmbd).squeeze()
			# fux = (fux@lmbd).squeeze()
		return xxp, fx, fu, fxx, fuu, fux



def tensorCont(P,a):
	
	# Tensor contraction function
	a = a.squeeze()
	T = np.zeros(P.shape[:-1])
	for i in range(P.shape[-1]):
		T += P[:,:,i]*a[i]
	return T
