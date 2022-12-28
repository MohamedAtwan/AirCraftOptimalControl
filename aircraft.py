import numpy as np 

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
		self.ns = 8
		self.ni = 2


	def vel_vector(self,xx):
		'''
			This function calculates the velocicty value and the gradients of the velocity w.r.t. the system states
			Inputs: xx --> states
			Outputs: V --> velocity value,  dV_x --> gradients of the velocity w.r.t the system states
		'''

		# Velocity value
		V = (xx[1,0]+xx[3,0])/(np.cos(xx[6,0])-np.sin(xx[6,0]))

		# Gradients
		dV_x = np.zeros((ns,1))
		dV_x[1,0] = 1/(np.cos(xx[6,0])-np.sin(xx[6,0]))
		dV_x[3,0] = 1/(np.cos(xx[6,0])-np.sin(xx[6,0]))
		dv_x[6,0] = (-(xx[1,0]+xx[3,0])*(-np.sin(xx[6,0])-np.cos(xx[6,0])))/(np.cos(xx[6,0])-np.sin(xx[6,0]))**2

		return V, dV_x


	def dragForce(self, xx):
		'''
			This function calculates the drag force value and the gradients of this force w.r.t. the system states
			Inputs: xx --> system states
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
		'''
		
		# parameters
		rho = self.rho
		S = self.S
		Cd0 = self.cd0
		Cda = self.cda
		V, dv_x = self.vel_vector(xx)
		alpha = xx[4,0]-xx[6,0]

		# Drag force
		D = 0.5 * rho * (V**2) * S *(Cd0 + Cda * alpha**2)

		# Gradients
		dD_x = np.zeros((self.ns,1))
		dD_x[1,0] = rho*V*dV_x[1,0]*S*Cd0
		dD_x[3,0] = rho*V*dV_x[3,0]*S*Cd0
		dD_x[4,0] = rho*V*S*Cda*alpha
		dD_x[6,0] = rho*V*dV_x[6,0]*S*(Cd0 + Cda * alpha**2) + rho*(V**2)*S*C*Cda*alpha

		return D,dD_x

	def liftForce(self,xx):
		'''
			This function calculates the lift force value and the gradients of this force w.r.t. the system states
			Inputs: xx --> system states
			Outputs: D --> Lift force value,  dD_x --> gradients of the lift force w.r.t the system states
		'''

		# parameters
		rho = self.rho
		V, dV_x = self.vel_vector(xx)
		alpha = xx[4,0]-xx[6,0]

		# Lift Force
		L = 0.5*rho*V**2*S*Cla*alpha

		# Gradients
		dL_x = np.zeros((self.ns,1))
		dL_x[1,0] = rho*V*dV_x[1,0]*S*Cla*alpha
		dL_x[3,0] = rho*V*dV_x[3,0]*S*Cla*alpha
		dL_x[4,0] = 0.5*rho*(V**2)*S*Cla
		dL_x[6,0] = rho*V*dV_x[6,0]*S*Cla*alpha + 0.5*rho*(V**2)*S*Cla

		return L, dL_x

	def acc_vector(xx,uu):

		'''
			This function calculates the acceleration value based on the forces acting on the aircraft like gravity, input thrust force, and drag force.
			 Also, it computes the gradients of this value w.r.t. the system states, and the system inputs
			Inputs: xx --> system states,  uu--> system Inputs
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
		'''

		# parameters
		m = self.m
		g = self.g 
		D, dD_x = self.dragForce(xx)
		alpha = xx[4,0]-xx[6,0]

		# Acceleration value from newton law
		Vd = (1/m) * (-D-*m*g*sin(xx[6,0]) + uu[0,0]*np.cos(alpha))

		# Gradients w.r.t the states
		dVd_x = np.zeros((ns,1))
		dVd_x[3,0] = -dD_x[3,0]/m 
		dVd_x[4,0] = -dD_x[4,0]/m - (uu[0,0]/m)*np.sin(alpha)
		dVd_x[6,0] = -dD_x[6,0]/m - (uu[0,0]/m)*np.sin(alpha)

		# Gradients w.r.t. the inputs
		dVd_u = np.zeros((ni,1))
		dVd_u[0,0] = (1/m)*np.cos(alpha)

		return Vd, dVd_x, dVd_u

	def step(self,xx,uu):
		'''
			This function takes on step discrete time step based on the current state and the current input
			Inputs: xx --> system states,  uu--> system Inputs
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
		'''
		xx = xx[:,None]
  		uu = uu[:,None]

  		# parameters
  		V, dV_x = self.vel_vector(xx)
  		Vd, dVd_x, dVd_u = self.acc_vector(xx,uu)
  		# L,dL_x = self.liftForce(xx,uu)
  		
  		# Auxiliary Variables
  		alpha = xx[4,0]-xx[6,0]
  		grad_alpha = np.transpose(np.array([[0,0,0,0,1,0,-1,0]]))
  		alpha_dot = xx[5,0]-xx[7,0]
  		grad_alpha_dot = np.transpose(np.array([[0,0,0,0,0,1,0,-1]]))

		xxp = np.zeros((ns,))
		xxp[0] = xx[0,0] + dt * xx[1,0]
		xxp[1] = xx[1,0] + dt * (Vd*np.cos(xx[6,0])-V*xx[7,0]*np.sin(xx[6,0]))

		xxp[2] = xx[2,0] + dt * xx[3,0]
		xxp[3] = xx[3,0] + dt * (-1*(Vd/mm)*np.sin(xx[6,0])-V*xx[7,0]*np.cos(xx[6,0]))

		xxp[4] = xx[4,0] + dt * xx[5,0]
		xxp[5] = xx[5,0] + dt * (uu[1,0]/J)

		xxp[6] = xx[6,0] + dt * xx[7,0]
		# xxp[7] = xx[7,0] + dt *((1/(m*V))*dL_x.T@np.transpose(np.atleast2d(xxp))+(g/V)*xx[7,0]*np.sin(x[6,0])+(uu[0,0]/(m*V))*alpha_dot*np.cos(alpha)-(Vd/V)*xx[7,0])
		xxp[7] = xx[7,0] + dt * ((1/m) * (rho*Vd*S*Cla*alpha + 0.5*rho*V*S*Cla*alpha_dot+(uu[0,0]/V)*alpha_dot*np.cos(alpha))+(g/V)*xx[7,0]*np.sin(x[6,0]) - (Vd/V)*xx[7,0])
		
		# Gradients
		fx = np.zeros((ns, ns))
		fu = np.zeros((ni, ns))

		# df1
		# Gradients of f1 w.r.t. the states
		fx[0,0] = 1
		fx[1,0] = dt

		# df2
		# Gradients of f2 w.r.t. the states
		for i in range(0,8):
			fx[i,1] = dt*(dVd_x[i,0]*np.cos(xx[6,0])-dV_x[i,0]*np.sin(xx[6,0]))
		fx[1,1] +=1

		# df3
		# Gradients of f3 w.r.t. the states
		fx[2,2] = 1
		fx[3,2] = dt

		# df4
		# Gradients of f4 w.r.t. the states
		for i in range(0,8):
			fx[i,3] = dt*(-1*(dVd_x[i,0]/mm)*np.sin(xx[6,0])-dV_x[i,0]*xx[7,0]*np.cos(xx[6,0]))
		fx[3,3] +=1

		# df5
		# Gradients of f5 w.r.t. the states
		fx[4,4] = 1
		fx[5,4] = dt

		# df6
		# Gradients of f6 w.r.t. the states
		fx[5,5] = 1

		# df7
		# Gradients of f7 w.r.t. the states
		fx[6,6] = 1
		fx[7,6] = dt

		# df8
		# Gradients of f8 w.r.t. the states
		for i in range(0,8):
			fx[i,7] = dt * ((1/m) * (rho*dVd_x[i,0]*S*Cla*alpha+rho*Vd*S*Cla*grad_alpha[i,0] +
					  0.5*rho*dV_x[i,0]*S*Cla*alpha_dot+0.5*rho*V*S*Cla*grad_alpha_dot[i,0]+
					  (((uu[0,0]*dV_x[i,0])/V**2)*alpha_dot+((uu[0,0]/V)*grad_alpha_dot[i,0]))*np.cos(alpha)-(uu[0,0]/V)*alpha_dot*grad_alpha[i,0]*np.sin(alpha))+
					  ((uu[0,0]*dV_x[i,0])/V**2)*xx[7,0]*np.sin(x[6,0]) - ((dVd_x[i,0]*V-dV_x[i,0]*Vd)/V**2)*xx[7,0])
		fx[7,7] += 1

		# dfu
		# Gradients w.r.t. the inputs
		fu[0,1] = dt * (dVd_u[0,0]*np.cos(xx[6,0]))
		fu[1,1] = dt * (dVd_u[1,0]*np.cos(xx[6,0]))
		
		fu[0,3] = dt * (-1*(dVd_u[0,0]/mm)*np.sin(xx[6,0]))
		fu[1,3] = dt * (-1*(dVd_u[1,0]/mm)*np.sin(xx[6,0]))
		
		fu[1,5] = dt/J
		
		fu[0,7] = xxp[7] = dt * ((1/m) * (rho*dVd_u[0,0]*S*Cla*alpha +(1/V)*alpha_dot*np.cos(alpha))- (dVd_u[0,0]/V)*xx[7,0])
		fu[1,7] = xxp[7] = dt * ((1/m) * (rho*dVd_u[1,0]*S*Cla*alpha - (dVd_u[1,0]/V)*xx[7,0])