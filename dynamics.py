import numpy as np
from math import log
from scipy.optimize import fsolve, least_squares, minimize, NonlinearConstraint

def round_theta(th):
	''' Some recursive algorithm to round theta to be between zero and 2pi'''
	if abs(th) < np.pi*2:
		return th
	else:
		if th < 0:
			return round_theta(th + np.pi*2)
		else:
			return round_theta(th - np.pi*2)

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
		self.QQt = np.eye(self.ns)
		self.QQt[0,0] = 1
		self.QQt[2,2] = 100
		# self.QQt[4,4] = 1e-1
		self.RRt = 1e-2*np.eye(self.ni)
		self.QQT = 10*self.QQt
		self.Temp = None
		self.eps_init = 1.5
		self.eps_end = 0.1
		self.speedLimit = 480
		self.epsilon = self.eps_init

	def get_equilibrium(self,theta,x,z):
		# TBD
		''' Calculate the equillibrium corresoponding to some (Theta, X, Z) values
			Inputs: theta, X, V
			outputs: State values vector "xx"
		'''
		m = self.m
		g = self.g
		rho = self.rho
		Cla = self.cla
		S = self.S
		xx = np.zeros((self.ns))
		xx[0] = x
		xx[1] = ((2*g)/(rho*S*Cla*theta))**0.5
		xx[2] = z
		xx[3] = 0 
		xx[4] = theta
		xx[5] = 0
		return xx


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

	def step(self,xx,uu):
		# X, Z, V, THETA, Q, GAMMA
		# print(uu)
		# sign = -1 if xx[1] < 0 else 1
		# xx[1] = np.min((abs(xx[1]),480))*sign
		# sign = -1 if xx[3] < 0 else 1
		# xx[3] = np.min((abs(xx[3]),480))*sign
		'''
			This function takes on step discrete time step based on the current state and the current input
			Inputs: xx --> system states,  uu--> system Inputs
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
		'''
		ns, ni = self.ns, self.ni
		xx = xx.reshape(-1,1)
		uu = uu.reshape(-1,1)

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

		dV_x = np.zeros((8,1))
		V, dV_x = self.vel_vector(xx)
		gamma, d_gamma = self.gamma_angle(xx)
		alpha = xx[3,0] - xx[5,0]
		D, dD_x = self.dragForce(xx)
		L,dL_x = self.liftForce(xx)

		xxp = np.zeros((ns,),dtype = np.float32)
		xxp[0] = xx[0,0] + dt*xx[2,0]*np.cos(xx[5,0])
		xxp[1] = xx[1,0] - dt*xx[2,0]*np.sin(xx[5,0])

		xxp[2] = xx[2,0] + (dt/m) * (-D - m*g*np.sin(xx[5,0]) + uu[0,0] * np.cos(alpha))
		xxp[3] = xx[3,0] + dt*xx[4,0]

		xxp[4] = xx[4,0] + dt * (uu[1,0]/J)
		xxp[5] = xx[5,0] + (dt/(m*xx[2,0]))*(L-m*g*np.cos(xx[5,0])+uu[0,0]*np.sin(alpha))

		# Gradients
		fx = np.zeros((ns, ns))
		fu = np.zeros((ni, ns))

		# df1
		# Gradients of f1 w.r.t. the states
		fx[0,0] = 1
		fx[2,0] = dt*np.cos(xx[5,0])
		fx[5,0] = -dt*xx[2,0]*np.sin(xx[5,0])

		# df2
		# Gradients of f2 w.r.t. the states
		fx[1,1] = 1
		fx[2,1] = -dt*np.sin(xx[5,0])
		fx[5,1] = -dt*xx[2,0]*np.cos(xx[5,0])

		# df3
		# Gradients of f3 w.r.t. the states
		for i in range(0,ns):
			fx[i,1] = (dt/m)*(-dD_x[i,0])
		
		fx[2,2] +=1
		fx[3,2] += (dt/m)*(-uu[0,0]*np.sin(alpha))
		fx[5,2] += (dt/m)*(-m*g*np.cos(xx[5,0])+uu[0,0]*np.sin(alpha))
		

		# df4
		# Gradients of f4 w.r.t. the states
		fx[3,3] = 1
		fx[4,3] = dt


		

		# df5
		# Gradients of f5 w.r.t. the states
		fx[4,4] = 1

		# df6
		# Gradients of f6 w.r.t. the states
		fx[2,5] = (-dt/(m*xx[2,0]**2))*L + (dt/(m*xx[2,0]))*dL_x[2,0]
		fx[3,5] = (dt/(m*xx[2,0]))*(dL_x[3,0] + uu[0,0]*np.cos(alpha))
		fx[5,5] = 1+(dt/(m*xx[2,0]))*(dL_x[5,0] + m*g*np.sin(xx[5,0]) - uu[0,0]*np.cos(alpha))

		# dfu
		# Gradients w.r.t. the inputs
		fu[0,2] = (dt/m)*np.cos(alpha)
		
		fu[0,5] = (dt/(m*xx[2,0]))*np.sin(alpha)
		
		fu[1,4] = dt/J

		return xxp, fx, fu

	def get_input(self,xx):
		''' Still work under progress'''
		V, dV_x = self.vel_vector(xx)
		gamma, d_gamma = self.gamma_angle(xx)
		D, dD_x = self.dragForce(xx)
		L,dL_x = self.liftForce(xx)

	def __quasi_cost(self,x):
		''' Still work under progress'''
		m = self.m
		rho = self.rho
		s = self.S
		cd0 = self.cd0
		cda = self.cda
		cla = self.cla
		g = self.g
		J = self.J
		dt = self.dt
		xx = self.Temp

		gamma, _ = self.gamma_angle(xx)
		D, dD_x = self.dragForce(xx)
		L,dL_x = self.liftForce(xx)
		cost = (x[0]*np.cos(xx[4]) - D * np.cos(gamma) - L*np.sin(gamma))**2 + ((x[0]*np.sin(xx[4]) - D * np.sin(gamma) + L*np.cos(gamma)) - m*g)**2
		return [cost]


	def get_quasi_static_trajectory(self,xx):
		''' Still work under progress'''
		N = xx.shape[1]
		uu = np.zeros((self.ni,N))
		xx_new = xx.copy()
		xx_new[4,:] = 20*np.pi/180
		x_init = [200]
		for tt in range(N):
			self.Temp = xx_new[:,tt].copy()
			sol = fsolve(self.__quasi_cost,x_init)
			uu[0,tt] = sol[0]
			# th = round_theta(sol[1])
			# xx_new[4,tt] = th
			x_init = [sol[0]]
			# print(self.__quasi_cost(x_init))
		return xx_new,uu



	def __point_constraints(self,x):
		''' Still work under progress'''
		m = self.m
		rho = self.rho
		s = self.S
		cd0 = self.cd0
		cda = self.cda
		cla = self.cla
		g = self.g
		J = self.J
		dt = self.dt

		xx = np.zeros((ns,1))
		xx[1,0] = x[0]
		xx[3,0] = x[1]
		xx[4,0] = x[2]
		gamma, _ = self.gamma_angle(xx)
		cost = [
				(x[3]*np.cos(x[2]) - D * np.cos(gamma) - L*np.sin(gamma)),
				(x[3]*np.sin(x[2]) - D * np.sin(gamma) + L*np.cos(gamma)) - m*g]
		return cost

	def gain_attitude_points(self,Theta_init, Theta_final, x_init, z_init, T, time_period):
		''' Still work under progress'''
		p1 = [0.1, 0.15]
		xx = np.zeros((2,8))
		for i, Theta in enumerate([Theta_init, Theta_final]):
			# random_floats = np.random.rand(2)
			res = minimize(self.__point_constraints, np.array([20*np.pi/180,180]), bounds = ((-40*np.pi,-180),(40*np.pi/180,480)))
			print(res.x)

			# gamma = np.rad2deg(res.x[0])
			# V = np.rad2deg(res.x[1])
			gamma = res.x[0]
			V = res.x[1]

			xx[i,1] = V*np.cos(gamma)
			xx[i,3] = -V*np.sin(gamma)
			xx[i,4] = Theta
			xx[i,6] = gamma
			if i == 0:
				xx[i,0] = x_init
				xx[i,2] = z_init
			else:
				xx[i,0] = x_init+time_period*xx[i,1]
				xx[i,2] = z_init+time_period*xx[i,3]
		return xx



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

		# bar, dbar = self.barrier_function(xx,uu)

		ll = 0.5*(xx - xx_ref).T@(QQt@(xx - xx_ref)) + 0.5*(uu - uu_ref).T@(RRt@(uu - uu_ref))# + bar

		lx = (QQt@(xx - xx_ref))#+dbar
		lu = RRt@(uu - uu_ref)
		return ll, lx, lu
	def barrier_function(self,xx,uu):
		''' Still work under progress'''
		rho = self.rho
		Cla = self.cla
		m = self.m
		g = self.g
		S = self.S
		F = (2*(-uu[0,0]*np.sin(xx[4,0])+m*g))/(rho*S*Cla*xx[4,0])
		fun = self.epsilon*(-np.real(np.emath.log(xx[1,0]**2-F)))#*(xx[1,0]-180)*(xx[3,0]-180)))
		dxFun = np.zeros((self.ns,1))
		dxFun[1,0] = -self.epsilon*(2*xx[1,0])/(xx[1,0]**2-F)
		# dxFun[3,0] = -self.epsilon/(-xx[1,0]+self.speedLimit)
		return fun, dxFun

	def update_epsilon(self,r):
		self.epsilon = (self.eps_init-self.eps_end)*r+self.eps_end

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

		return llT, lTx