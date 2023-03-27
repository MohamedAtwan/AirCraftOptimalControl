import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) #minor grid
# from acrobatic_newton import sigmoid_fcn, reference_position

class Airfoil:
	def __init__(self,th,xx_star, xx_ref,dt = 1e-3,xlim = [0,15], ylim = [-4,4]):
		self.th = th
		self.xlim = xlim
		self.ylim = ylim
		self.airfoil = self.__create_airfoil()
		self.xx_star = xx_star
		self.xx_ref = xx_ref
		self.dt = dt

	def update_pose(self, theta, x_loc,y_loc):
		v = self.airfoil
		T = np.array([[np.cos(theta), -np.sin(theta), -x_loc],
					  [np.sin(theta), np.cos(theta), y_loc],
					  [0, 0, 1]])
		v = T@v
		# plt.plot(-v[0,:],v[1,:])
		# plt.xlim([-self.xlim, self.xlim])
		# plt.ylim([-self.ylim, self.ylim])
		# plt.show()
		return v

	def run_animation(self):
		xx_star = self.xx_star
		xx_ref = self.xx_ref
		dt = self.dt
		tf = 1.0
		TT = int(tf/dt)
		time = np.linspace(0,tf,TT)

		fig = plt.figure()
		ax = fig.add_subplot(111, autoscale_on=False, xlim=(self.xlim[0], self.xlim[1]), ylim=(self.ylim[0], self.ylim[1]))
		ax.grid()
		# no labels
		ax.set_yticklabels([])
		ax.set_xticklabels([])


		self.line0, = ax.plot([], [], 'o-', lw=2, c='b', label='Optimal')
		self.line1, = ax.plot([], [], '*-', lw=2, c='g',dashes=[2, 2], label='Reference')

		self.time_template = 't = %.1f s'
		self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
		fig.gca().set_aspect('equal', adjustable='box')

		# Subplot
		left, bottom, width, height = [0.64, 0.13, 0.2, 0.2]
		ax2 = fig.add_axes([left, bottom, width, height])
		ax2.xaxis.set_major_locator(MultipleLocator(2))
		ax2.yaxis.set_major_locator(MultipleLocator(0.25))
		ax2.set_xticklabels([])


		ax2.grid(which='both')
		ax2.plot(time, xx_star[1,:],c='b')
		ax2.plot(time, xx_ref[1,:], color='g', dashes=[2, 1])

		self.point1, = ax2.plot([], [], 'o', lw=2, c='b')
		ani = animation.FuncAnimation(fig, self.animate, TT, interval=1, blit=True, init_func=self.anime_init)
		writervideo = animation.PillowWriter(fps = 5)
		ani.save('Figures/AircraftBehavior.gif',writer = writervideo)
		ax.legend(loc="lower left")
		plt.show()

	def anime_init(self):
		self.line0.set_data([], [])
		self.line1.set_data([], [])

		self.point1.set_data([], [])

		self.time_text.set_text('')
		return self.line0, self.line1, self.time_text, self.point1


	def animate(self,i):
		xx_star = self.xx_star
		xx_ref = self.xx_ref
		dt = self.dt
		v = self.update_pose(xx_star[3,i], xx_star[0,i], xx_star[1,i])
		# Trajectory
		# thisx0 = [0, np.sin(xx_star[1, i])]
		# thisy0 = [0, np.cos(xx_star[1, i])]
		self.line0.set_data(-v[0,:], v[1,:])

		# Reference
		# thisx1 = [0, np.sin(xx_ref[1, -1])]
		# thisy1 = [0, np.cos(xx_ref[1, -1])]
		vr = self.update_pose(xx_ref[3,-1], xx_ref[0,-1], xx_ref[1,-1])
		self.line1.set_data(-vr[0,:], vr[1,:])

		self.point1.set_data(i*dt, xx_star[1, i])

		self.time_text.set_text(self.time_template % (i*dt))
		return self.line0, self.line1, self.time_text, self.point1

	def __create_airfoil(self):
		th = self.th
		t = th/100.0
		x = np.linspace(0,1,100)
		yt = 5*t*(0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 +  0.2843*x**3 - 0.1015*x**4)

		r = 1.1019*t**2

		xU, xL = x.copy(), x.copy()
		yU, yL = yt.copy(), -yt.copy()

		xx = np.concatenate([xU,xL], axis = 0)
		yy = np.concatenate([yU,yL], axis = 0)

		v = np.concatenate([np.atleast_2d(xx),
							np.atleast_2d(yy),
							np.ones((1,yy.shape[0]))], axis = 0)
		return v

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
  slope = tt.shape[0]*0.1
  TT = tt.shape[0]
  pp = np.zeros((TT,))
  vv = np.zeros((TT,))
  pp[:TT//2] = p0+sigmoid_fcn(tt[:TT//2] - tt[TT//2]/2,slope)[0]*(pT - p0)
  vv[:TT//2] = sigmoid_fcn(tt[:TT//2] - tt[TT//2]/2,slope)[1]*(pT - p0)
  pp[TT//2:] = p0+sigmoid_fcn(-tt[:TT//2] + tt[TT//2]/2,slope)[0]*(pT - p0)
  vv[TT//2:] = sigmoid_fcn(-tt[:TT//2] + tt[TT//2]/2,slope)[1]*(pT - p0)
  return pp, vv

# if __name__ == '__main__':
# 	# plot_airfoil(30, -45*np.pi/180.0, 2, 2)
# 	xx_star = np.load(f'Data/xx_star_acrobatic.npy')
# 	tf = 1.0
# 	dt = 1e-3
# 	ns = 6
# 	ni = 2
# 	TT = int(tf/dt)
# 	tt = np.linspace(0,tf,TT)
	

# 	xx_ref = np.zeros((ns, TT))
# 	uu_ref = np.zeros((ni, TT))

# 	x0,z0,alpha0 = 0,0,6*np.pi/180
# 	xf,zf,alphaf = 10,2,6*np.pi/180
# 	vz = (zf-z0)/tf



# 	zz,zzd = reference_position(tt, z0, zf)
# 	# xx,xxd = reference_position(tt, x0, xf)

# 	# xxe,uue = dyn.get_equilibrium(np.zeros(dyn.ns,),tt)
# 	xx_ref[0,:] = x0+((xf-x0)/tf)*tt
# 	xx_ref[1,:] = zz.copy()
# 	# for i in range(2,dyn.ns):
# 	#   xx_ref[i,:] = xxe[i]#(zzd**2+((xf-x0)/tf)**2)**0.5
# 	aircraft = Airfoil(30,xx_star,xx_ref)
# 	aircraft.run_animation()
