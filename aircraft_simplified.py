import numpy as np
from scipy.optimize import fsolve, least_squares, minimize, NonlinearConstraint

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
		self.QQt[0,0] = 10
		self.QQt[2,2] = 1000
		self.QQt[4,4] = 10
		self.RRt = 1e-1*np.eye(self.ni)
		self.QQT = 10*self.QQt
		self.Temp = None

	def gamma_angle(self,xx):
		# print(xx[1,0])
		ns = self.ns
		gamma = np.math.atan2(-xx[3,0],xx[1,0])
		d_gamma = np.zeros((ns,1))
		val = -xx[3,0]/xx[1,0]
		d_gamma[1,0] = (1/(1+val**2))*(xx[3,0]/xx[1,0]**2)
		d_gamma[3,0] = (1/(1+val**2))*(-1/xx[1,0])
		return gamma, d_gamma


	def vel_vector(self,xx):
		'''
			This function calculates the velocicty value and the gradients of the velocity w.r.t. the system states
			Inputs: xx --> states
			Outputs: V --> velocity value,  dV_x --> gradients of the velocity w.r.t the system states
		'''

		xx = xx.reshape(-1,1)

		# Parameters
		ns = self.ns
		gamma, _ = self.gamma_angle(xx)
		dV_x = np.zeros((ns,1))

		if np.cos(gamma)==np.sin(gamma):
			# Velocity value
			if np.cos(gamma) == 0:
				V = xx[3,0]
				# Gradients
				dV_x[3,0] = 1
			else:
				V = xx[1,0]/np.cos(gamma)
				# Gradients
				dV_x[1,0] = 1/np.cos(gamma)
		else:
			# Velocity value
			V = (xx[1,0]+xx[3,0])/(np.cos(gamma)-np.sin(gamma))
			# Gradients
			dV_x[1,0] = 1/(np.cos(gamma)-np.sin(gamma))
			dV_x[3,0] = 1/(np.cos(gamma)-np.sin(gamma))

		return V, dV_x


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
		V, dV_x = self.vel_vector(xx)
		gamma, _ = self.gamma_angle(xx)
		alpha = xx[4,0]-gamma
		# Drag force value
		D = 0.5 * rho * (V**2) * S *(Cd0 + Cda * alpha**2)

		# Gradients
		dD_x = np.zeros((self.ns,1))
		dD_x[1,0] = rho*V*dV_x[1,0]*S*Cd0
		dD_x[3,0] = rho*V*dV_x[3,0]*S*Cd0
		dD_x[4,0] = rho*V*S*Cda*alpha

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
		V, dV_x = self.vel_vector(xx)
		gamma, _ = self.gamma_angle(xx)
		alpha = xx[4,0]-gamma

		# Lift Force value
		L = 0.5*rho*V**2*S*Cla*alpha

		# Gradients
		dL_x = np.zeros((self.ns,1))
		dL_x[1,0] = rho*V*dV_x[1,0]*S*Cla*alpha
		dL_x[3,0] = rho*V*dV_x[3,0]*S*Cla*alpha
		dL_x[4,0] = 0.5*rho*(V**2)*S*Cla

		return L, dL_x

	def step(self,xx,uu):
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

		V = 2.0
		dV_x = np.zeros((8,1))
		V, dV_x = self.vel_vector(xx)
		gamma, d_gamma = self.gamma_angle(xx)
		D, dD_x = self.dragForce(xx)
		L,dL_x = self.liftForce(xx)

		xxp = np.zeros((ns,),dtype = np.float32)
		xxp[0] = xx[0,0] + dt * xx[1,0]
		xxp[1] = xx[1,0] + (dt/m)*(uu[0,0]*np.cos(xx[4,0]) - D * np.cos(gamma) - L*np.sin(gamma))

		xxp[2] = xx[2,0] + dt * xx[3,0]
		xxp[3] = xx[3,0] + (dt/m)*(uu[0,0]*np.sin(xx[4,0]) - D * np.sin(gamma) + L*np.cos(gamma)) - m*g

		xxp[4] = xx[4,0] + dt * xx[5,0]
		xxp[5] = xx[5,0] + dt * (uu[1,0]/J)

		# Gradients
		fx = np.zeros((ns, ns))
		fu = np.zeros((ni, ns))

		# df1
		# Gradients of f1 w.r.t. the states
		fx[0,0] = 1
		fx[1,0] = dt

		# df2
		# Gradients of f2 w.r.t. the states
		for i in range(0,ns):
			fx[i,1] = (dt/m)*(-dD_x[i,0]*np.cos(gamma)+D*d_gamma[i,0]*np.sin(gamma)-dL_x[i,0]*np.sin(gamma)-L*d_gamma[i,0]*np.cos(gamma))
			if i == 4:
				fx[i,1] += ((dt/m)*uu[0,0]*np.sin(xx[4,0]))
		fx[1,1] +=1

		# df3
		# Gradients of f3 w.r.t. the states
		fx[2,2] = 1
		fx[3,2] = dt

		# df4
		# Gradients of f4 w.r.t. the states
		for i in range(0,ns):
			fx[i,1] = (dt/m)*(-dD_x[i,0]*np.sin(gamma)+D*d_gamma[i,0]*np.cos(gamma)+dL_x[i,0]*np.cos(gamma)-L*d_gamma[i,0]*np.sin(gamma))
			if i == 4:
				fx[i,1] += (-(dt/m)*uu[0,0]*np.cos(xx[4,0]))
		fx[3,3] +=1

		# df5
		# Gradients of f5 w.r.t. the states
		fx[4,4] = 1
		fx[5,4] = dt

		# df6
		# Gradients of f6 w.r.t. the states
		fx[5,5] = 1

		# dfu
		# Gradients w.r.t. the inputs
		fu[0,1] = (dt/m)*np.cos(xx[4,0])
		
		fu[0,3] = (dt/m)*np.sin(xx[4,0])
		
		fu[1,5] = dt/J

		return xxp, fx, fu


	def __point_constraints(self,x, *args):
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
				args[0]-dt*x[0],
				(x[3]*np.cos(x[2]) - D * np.cos(gamma) - L*np.sin(gamma)),
				args[1]-dt*x[1]
				(x[3]*np.sin(x[2]) - D * np.sin(gamma) + L*np.cos(gamma)) - m*g]
		return cost

	def gain_attitude_points(self,Theta_init, Theta_final, x_init, z_init, T, time_period):
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

		ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

		lx = QQt@(xx - xx_ref)
		lu = RRt@(uu - uu_ref)

		return ll, lx, lu

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

	def phase_data(self,dt,xx):
		x_max, x_min