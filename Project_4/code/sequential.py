import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

display_progress = True

def jacobi(grid, f, dx):
    newgrid = np.zeros(shape=grid.shape)

    # apply evolution operator
    newgrid[1:-1,1:-1] = 0.25 * (grid[1:-1,:-2] + grid[1:-1,2:] +
                                 grid[:-2,1:-1] + grid[2:,1:-1] - f[1:-1,1:-1]*dx**2)
                    

    # copy boundary conditions
    newgrid[0,:]  = grid[0,:]
    newgrid[-1,:] = grid[-1,:]
    newgrid[:,0]  = grid[:,0]
    newgrid[:,-1] = grid[:,-1]
    
    return newgrid
    
    
def initgrid(gridsize):
    x = np.zeros((gridsize,gridsize))
    # x = np.random.randn(gridsize,gridsize)
    x[0,:]  = 1
    x[:,-1] = 1
    x[-1,:] = 1
    x[:,0]  = 1
    
    return x
    
    

n = 100  # grid-size
L = 1.0    # square bdy length
epsilon = 1e-8
dx = L/(n-1)   # FDM step size
u = initgrid(n)   # initial grid, n by n np array
x = np.linspace(0, L, n)
X, Y = np.meshgrid(x, x)  # X, Y grid
f = np.sin(X-Y)    # RHS values, n by n np array

# jacobian method, in each iteration: information exchange + jacobian iteration
t = -timer()
iter = 0
while True:
    if iter % 1000 == 0:   # every 1000 iterations, check if convergence is reached
        u_old = u
    u = jacobi(u, f, dx)
    if iter % 1000 == 0:
        diff = np.absolute(u_old - u).max()
        if display_progress:
            print("max at iteration ", iter, " is: ", diff)
        if diff < epsilon:
            break
    iter = iter + 1
    
t += timer()

print("Global elapsed run time is: ", t, " seconds")




