import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

######################################
# Define the desired velocity profile
######################################

def sigmoid_fcn(tt):
  """
    Sigmoid function

    Return
    - s = 1/1+e^-t
    - ds = d/dx s(t)
  """

  ss = 1/(1 + np.exp(-tt))

  ds = ss*(1-ss)

  return ss, ds

def reference_position(tt, p0, pT, T):
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

  pp = p0 + sigmoid_fcn(tt - T/2)[0]*(pT - p0)
  vv = sigmoid_fcn(tt - T/2)[1]*(pT - p0)

  return pp, vv


# a function to fix the input in a certain time
def u_func(tt, Tmax):
    # y = 10-(10*tt/(Tmax))+1
    y = (10*tt/(Tmax))
    # y = np.ones(len(tt))*0
    return y

def constraints(x, *args):
    tt = args[0]
    theta = args[1]
    T = args[2]
    m = 12
    rho = 1.2
    s=0.61
    cd0 = 0.1716
    cda = 2.395
    cla = 3.256
    g=9.81
    v = np.sqrt(x[0]**2 + x[1]**2)
    v_d = (x[0]*x[1] + x[2]*x[3])/v
    alfa = theta - x[4]
    D = 0.5 * rho * v**2 * s * (cd0 + (cda * (alfa**2)))
    L = 0.5 * rho * v**2 * s * cla * alfa
    return [x[0] - v * np.cos(x[4]),
            x[1] + v_d * np.cos(x[4]) + v * x[5] * np.sin(x[4]),
            x[2] + v * np.sin(x[4]),
            x[3] + v * x[5] * np.cos(x[4]) + v_d * np.sin([x[4]]),
            m * v_d + D + m * g * np.sin(x[4]) - T * np.cos(alfa),
            m* v * x[5] - L + m * g * np.cos(x[4]) - T * np.sin(alfa)
             ]

######################################
# Main
######################################
tf = 1
dt = 1e-1
ns = 8
ni = 2
# TT = int(tf/dt)
TT = int(tf/dt)
NN = TT
print('-*-*-*-*-*-')

tt = np.linspace(0,tf,TT)
M = u_func(tt, tf*TT) #the input
T = u_func(tt, tf*TT) #the input


# print("M", M[0])
# print("T", T[0])
"""
plt.figure()
plt.plot(tt, M, 'b', linewidth=2)
"""
J=0.24

x0_in = np.zeros((ns, TT)) #the trajectory to be searched for
theta_2d = np.ones(TT)*M/J #from the constraints

for ii in range(TT):
    theta_d = theta_2d[ii] * ii
    theta = 0.5 * theta_2d[ii] * (ii**2) % 6.28
    print("iter", ii)
    root = fsolve(constraints, [100.0, 500.0, 200.0, 100.0, 0.1, 0.5], args=(ii, theta, T[ii]))
    print(root.shape)
    if np.any(np.isnan(root)):
        x0_in[0, ii] = x0_in[0, ii-1]
        x0_in[1, ii] = x0_in[1, ii-1]
        x0_in[2, ii] = x0_in[2, ii-1]
        x0_in[3, ii] = x0_in[3, ii-1]
        x0_in[4, ii] = x0_in[4, ii-1]
        x0_in[5, ii] = x0_in[5, ii-1]
        x0_in[6, ii] = x0_in[6, ii-1]
        x0_in[7, ii] = x0_in[7, ii-1]

    if not np.any(np.isnan(root)):
        x0_in[0, ii] = ii * np.sqrt(root[0]**2 + root[2]**2) * np.cos(root[4])
        x0_in[1, ii] = root[0]
        x0_in[2, ii] = (-ii) * np.sqrt(root[0]**2 + root[2]**2) * np.sin(root[4])
        x0_in[3, ii] = root[2]
        x0_in[4, ii] = theta
        x0_in[5, ii] = theta_d
        x0_in[6, ii] = root[4]
        x0_in[7, ii] = root[5]


print(x0_in[:, 0])
v = np.zeros((1,TT))
for jj in range(TT-1):
    v[0, jj] = np.sqrt(x0_in[0, jj]**2 + x0_in[2, jj]**2)



plt.figure()
plt.plot(tt, x0_in[0, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{0}(t)$'])

plt.figure()
plt.plot(tt, x0_in[1, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{1}(t)$'])

plt.figure()
plt.plot(tt, x0_in[2, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{2}(t)$'])

plt.figure()
plt.plot(tt, x0_in[3, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{3}(t)$'])

plt.figure()
plt.plot(tt, x0_in[4, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{4}(t)$'])

plt.figure()
plt.plot(tt, x0_in[5, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{5}(t)$'])


plt.figure()
plt.plot(tt, x0_in[6, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{6}(t)$'])

plt.figure()
plt.plot(tt, x0_in[7, :], 'b', linewidth=2)
plt.grid()
plt.legend(['$x_{7}(t)$'])



plt.show()



"""
p0 = 5 # ok for x and z
pT = 10 # ok for x and z
px_ref, vx_ref = reference_position(tt, p0, pT, tf)
py_ref, vy_ref = reference_position(tt, p0, pT, tf)
print("px ref {}, vx ref {}".format(px_ref, vx_ref))

tt_hor = np.linspace(0,tf,TT)
print("tt hor shape", tt_hor.shape)
print("px shape", px_ref.shape)
plt.figure()
# plt.plot(tt_hor, reference_position(tt_hor, p0, pT, TT)[0], 'b', linewidth=2)
plt.plot(tt_hor, px_ref, 'b', linewidth=2)
plt.grid()
plt.legend(['$p_{des}(t)$'])
plt.figure()
# plt.plot(tt_hor, reference_position(tt_hor, p0, pT, TT)[1], 'r' , linewidth=2)
plt.plot(tt_hor, vx_ref, 'r' , linewidth=2)
plt.grid()
plt.legend(['$v_{des}(t)$'])
plt.show()
"""