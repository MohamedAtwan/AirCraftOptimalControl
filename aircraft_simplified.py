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

	def get_equilibrium(self,x,z, theta,tt):
		# TBD
		''' Calculate the equillibrium corresoponding to some (Theta, X, Z) values
			Inputs: theta, X, V
			outputs: State values vector "xx"
		'''
		# alpha = 5*np.pi/180 # radians
		# alpha = -np.pi+(random()*2*np.pi)
		TT = tt.shape[0]
		xx = np.zeros((self.ns,TT))
		uu = np.zeros((self.ni,TT))
		xx[0,0] = x
		xx[1,0] = z
		xx[2,0] = 170
		xx[5,0] = 0
		xx[3,0] = theta
		uu[0,0] = (random()*100)
		# alpha = xx[3,0]-xx[5,0]
		self.Temp = xx[:,0].copy(), theta
		x_init = np.array([xx[2,0],xx[5,0],uu[0,0]])
		for i in range(TT-1):
			# sol = least_squares(self.equilibrium_cost,x_init,ftol=1e-3,bounds = ((0,-np.pi,0.0,-np.pi),(480,np.pi,100,np.pi)), max_nfev = 2000)
			sol = fsolve(self.equilibrium_cost,x_init)
			# return xx, np.array([uu,0])
			# if not(sol.success):
			# 	raise RunTimeError(sol.message)
			xx[2,i] = sol[0]
			xx[5,i] = round_theta(sol[1])
			# xx[3,i] = round_theta(sol[3])
			uu[0,i] = sol[2]
			xx[:,i+1] = self.step(xx[:,i],uu[:,i])[0]
			# xx[5,i+1] = round_theta(xx[5,i+1])
			x_init = np.array([xx[2,i+1],xx[5,i+1],uu[0,i]])
			# print(x_init)
			self.Temp = xx[:,i].copy(), theta
		return xx, uu
			
	def equilibrium_cost(self,x):
		m = self.m
		g = self.g
		tf = 1
		NN = int(tf/self.dt)
		xx = np.zeros((self.ns,NN))
		xx[:,0] = self.Temp[0].copy()
		theta = self.Temp[1]
		xx[2,0] = x[0]
		xx[5,0] = x[1]
		xx[3,0] = theta
		uu = x[2]
		alpha = xx[3,0]-xx[5,0]
		D,_ = self.dragForce(xx)
		L,_ = self.liftForce(xx)
		# cost = 0
		# for i in range(NN-1):
		# 	xx[:,i+1] = self.step(xx[:,i],np.array([uu,0]))[0].flatten()
		# 	cost += np.sum((xx[:,i+1]-xx[:,i])**2)

		cost = [-D - m*g*np.sin(xx[5,0]) + uu * np.cos(alpha),
				 L - m*g*np.cos(xx[5,0]) + uu * np.sin(alpha)]
				 # xx[3,0]-xx[5,0]-alpha]
		return [sum(cost),]*3
		# return -D - m*g*np.sin(xx[5]) + uu * np.cos(alpha)
		# cost = [-D - m*g*np.sin(xx[5]) + uu * np.cos(alpha),
		# 		 L - m*g*np.cos(xx[5]) + uu * np.sin(alpha),
		# 		 xx[2]*np.cos(xx[5]),
		# 		 -xx[2]*np.sin(xx[5])]
		# # return np.sum(cost)
		# cost = np.atleast_2d(cost)@np.atleast_2d(cost).T
		# return cost[0,0]




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
			Inputs: xx --> system states,  uu--> system Inputs
			Outputs: D --> Drag force value,  dD_x --> gradients of the drag force w.r.t the system states
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
			fx[i,2] = (dt/m)*(-dD_x[i,0])
		
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

		# Hessian of napla_xx

		fxx = np.zeros((self.ns,self.ns,self.ns))
		fxx1 = np.zeros((self.ns,self.ns))
		fxx2 = np.zeros((self.ns,self.ns))
		fxx3 = np.zeros((self.ns,self.ns))
		fxx4 = np.zeros((self.ns,self.ns))
		fxx5 = np.zeros((self.ns,self.ns))
		fxx6 = np.zeros((self.ns,self.ns))

		fxx1[2,5] = -dt*np.sin(xx[5,0])
		fxx1[5,2] = -dt*np.sin(xx[5,0])
		fxx1[5,5] = -dt*xx[2,0]*np.cos(xx[5,0])

		fxx2[2,5] = dt*np.cos(xx[5,0])
		fxx2[5,2] = -dt*np.cos(xx[5,0])
		fxx2[5,5] = dt*xx[2,0]*np.sin(xx[5,0])

		fxx3[2,2] = (dt/m)*-1*(rho*S*(Cd0+Cda*alpha**2))
		fxx3[2,3] = (dt/m)*-1*(rho*xx[2,0]*2*alpha)
		fxx3[2,5] = (dt/m)*(rho*xx[2,0]*2*alpha)
		fxx3[3,2] = (dt/m)*-1*(rho*2*xx[2,0]*S*Cda*alpha)
		fxx3[3,3] = (dt/m)*(-1*(rho*(xx[2,0]**2)*S*Cda)-uu[0,0]*np.cos(alpha))
		fxx3[3,5] = (dt/m)*((rho*(xx[2,0]**2)*S*Cda)+uu[0,0]*np.cos(alpha))
		fxx3[5,2] = (dt/m)*-1*(-rho*2*xx[2,0]*S*Cda*alpha)
		fxx3[5,3] = (dt/m)*(-1*(-rho*(xx[2,0]**2)*S*Cda)+uu[0,0]*np.cos(alpha))
		fxx3[5,5] = (dt/m)*((-rho*(xx[2,0]**2)*S*Cda)-uu[0,0]*np.cos(alpha)+m*g*np.sin(xx[5,0]))


		fxx6[2,3] = (-dt*0.5*rho*S*Cla)/m + (dt*rho*S*Cla)/m
		fxx6[2,5] = (dt*0.5*rho*S*Cla)/m - (dt*rho*S*Cla)/m
		fxx6[3,2] = (-dt/(m*xx[2,0]**2))*(dL_x[3,0] + uu[0,0]*np.cos(alpha)) + (rho*xx[2,0]*S*Cla)*(dt/(m*xx[2,0]))
		fxx6[3,3] = -uu[0,0]*np.sin(alpha)*(dt/(m*xx[2,0]))
		fxx6[3,5] = uu[0,0]*np.sin(alpha)*(dt/(m*xx[2,0]))
		fxx6[5,2] = (-dt/(m*xx[2,0]**2))*(dL_x[5,0] + m*g*np.sin(xx[5,0]) - uu[0,0]*np.cos(alpha))+(-rho*xx[2,0]*S*Cla)*(dt/(m*xx[2,0]))
		fxx6[5,3] = uu[0,0]*np.sin(alpha)*(dt/(m*xx[2,0]))
		fxx6[5,5] = (-uu[0,0]*np.sin(alpha)+m*g*np.cos(xx[5,0]))*(dt/(m*xx[2,0]))

		fxx[0,:,:] = fxx1
		fxx[1,:,:] = fxx2
		fxx[2,:,:] = fxx3
		fxx[5,:,:] = fxx6
		# Hessian napla_ux
		fux = np.zeros((self.ni,self.ns,self.ns))
		fux[0,3,2] = (dt/m)*(-np.sin(alpha))
		fux[0,5,2] = (dt/m)*(np.sin(alpha))
		fux[0,3,5] = (dt/(m*xx[2,0]))*(np.cos(alpha))
		fux[0,5,5] = (dt/(m*xx[2,0]))*(-np.cos(alpha))
		# fux[0,2,5] = (-dt/(m*xx[2,0]**2))*np.sin(alpha)

		# Hessian napla_uu
		fuu = np.zeros((self.ni,self.ni,self.ns))

		# Hessian napla_xu
		fxu = np.zeros((self.ns,self.ni,self.ns))
		fxu[3,0,2] = (dt/m)*(-np.sin(alpha))
		fxu[5,0,2] = (dt/m)*np.sin(alpha)
		fxu[3,0,5] = (dt/(m*xx[2,0]))*(np.cos(alpha))
		fxu[5,0,5] = (dt/(m*xx[2,0]))*(-np.cos(alpha))

		if args:
			# if lmbd.shape == xx.shape:
			fxx = (fxx@lmbd).squeeze()
			fuu = (fuu@lmbd).squeeze()
			fxu = (fxu@lmbd).squeeze()
			fux = (fux@lmbd).squeeze()
		return xxp, fx, fu, fxx, fuu, fux, fxu



	