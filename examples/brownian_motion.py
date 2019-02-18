import numpy as np
from pylab import show
from math import sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def brownian(x0, n, dt, delta, out=None):

   #  n : The number of steps to take.
   #  dt : time step
   #  delta : "speed" of motion
   #  out :If `out` is NOT None, it specifies the array in which to put the
   #      result.  If `out` is None, a new numpy array is created and returned.
    x0 = np.asarray(x0) #I.C
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt)) #generate n numbers for sample
    if out is None: #create out array
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out) #cumulative sum for random variables
    out += np.expand_dims(x0, axis=-1)#initial condition.
    return out

def main():

    fig = plt.figure(1) #prepare plot
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax = fig.add_subplot(2, 3, 3, projection='3d')

    delta = 2 # The Wiener process parameter.
    T = 10.0
    N = 500# Number of steps.
    dt = T/N
    m = 5 # Number of "lines"
    x = np.empty((m,N+1))# Create an empty array to store the realizations.
    x[:, 0] =  0# Initial values of x.

    brownian(x[:,0], N, dt, delta, out=x[:,1:])

    t = np.linspace(0.0, T, N+1)
    for i in range(m):
        ax1.plot(t, x[i])
    ax1.set_title('1D Brownian Motion')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.grid(True)

    ax2.plot(x[0],x[1])
    ax2.plot(x[0,0],x[1,0], 'go') #begin
    ax2.plot(x[0,-1], x[1,-1], 'ro') #end
    ax2.set_title('2D Brownian Motion')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.axis('equal')
    ax2.grid(True)

    xdata, ydata, zdata = x[:3,:]
    ax.plot3D(xdata, ydata, zdata)
    ax.set_title('3D Brownian Motion')


    show()
    return
main()