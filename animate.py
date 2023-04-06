import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator) #minor grid

class Airfoil:
	'''
		Calculates the NACA airfoil and animates it on a specified trajectory.
	'''
	def __init__(self,th,xx_star, xx_ref,dt = 1e-3,xlim = [0,15], ylim = [-4,4]):
		self.th = th
		self.xlim = xlim
		self.ylim = ylim
		self.airfoil = self.__create_airfoil()
		self.xx_star = xx_star
		self.xx_ref = xx_ref
		self.dt = dt

	def update_pose(self, theta, x_loc,y_loc):
		'''
			Updates the pose for the airfoil.

			Inputs: theta --> the angle about z-axis
					x_loc --> location in the x-direction
					y_loc --> location in the y-direction

			Outputs: v --> The new coordinates for every point on the airfoil
		'''
		v = self.airfoil
		T = np.array([[np.cos(theta), -np.sin(theta), -x_loc],
					  [np.sin(theta), np.cos(theta), y_loc],
					  [0, 0, 1]])
		v = T@v
		return v

	def run_animation(self,name = ''):
		'''
			Animates the airfoil based on the trajectory passed to the class constructor.

			Inputs: name --> specific name for the generated gif file
		'''
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
		ani = animation.FuncAnimation(fig, self.animate, TT, interval=1, blit=False, init_func=self.anime_init)
		writervideo = animation.PillowWriter(fps = 15)
		ani.save(f'Figures/AircraftBehavior_{name}.gif',writer = writervideo)
		ax.legend(loc="lower left")
		plt.show()

	def anime_init(self):
		'''
			Initializing the plot and the containers.
		'''
		self.line0.set_data([], [])
		self.line1.set_data([], [])

		self.point1.set_data([], [])

		self.time_text.set_text('')
		return self.line0, self.line1, self.time_text, self.point1


	def animate(self,i):
		'''
			updates the frame with the updated dynamics

			Inputs: i --> time instance
		'''
		xx_star = self.xx_star
		xx_ref = self.xx_ref
		dt = self.dt
		v = self.update_pose(xx_star[3,i], xx_star[0,i], xx_star[1,i])
		# Trajectory
		self.line0.set_data(-v[0,:], v[1,:])

		# Reference
		vr = self.update_pose(xx_ref[3,-1], xx_ref[0,-1], xx_ref[1,-1])
		self.line1.set_data(-vr[0,:], vr[1,:])

		self.point1.set_data(i*dt, xx_star[1, i])

		self.time_text.set_text(self.time_template % (i*dt))
		return self.line0, self.line1, self.time_text, self.point1

	def __create_airfoil(self):
		'''
			creates a symmetric airfoil based on NACA_XX model.

			Outputs: the coordinates for each point on the Airfoil.
		'''
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
