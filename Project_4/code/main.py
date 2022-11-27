import numpy as np
from mpi4py import MPI
from utilities import jacobi, initgrid, exchange_info
import matplotlib
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

display_progress = True
n = 200  # grid-size
L = 1.0    # square bdy length
epsilon = 1e-8
dx = L/(n-1)   # FDM step size
u, f = None, None
if rank == 0:
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x)  # X, Y grid
    f = np.sin(X-Y)    # RHS values, n by n np array
    u = initgrid(n)   # initial grid, n by n np array
    # print(u)
    # print(f)

# setting up local rows
sub_u = np.empty((int(n/size), n))
sub_f = np.empty((int(n/size), n))
comm.Scatterv(u, sub_u, root=0)
comm.Scatterv(f, sub_f, root=0)
if rank == 0:
    sub_u = np.vstack([sub_u, np.zeros(n)])
elif rank == size -1:
    sub_u = np.vstack([np.zeros(n), sub_u])
else:
    sub_u = np.vstack([np.zeros(n), sub_u, np.zeros(n)])


# jacobian method, in each iteration: information exchange + jacobian iteration
t = -MPI.Wtime()
iter = 0
while True:
    if iter % 1000 == 0:   # every 1000 iterations, check if convergence is reached
        u_old = sub_u
    exchange_info(rank, size, comm, sub_u)
    sub_u = jacobi(sub_u, sub_f, dx, rank, size)
    if iter % 1000 == 0:
        diff = np.absolute(u_old - sub_u).max()
        diff_all = comm.allreduce(diff, MPI.MAX)
        if rank == 0 and display_progress:
            print("max at iteration ", iter, " is: ", diff_all)
        if diff_all < epsilon:
            break
    iter = iter + 1
t += MPI.Wtime()
t_all = comm.allreduce(t, MPI.MAX)
if rank == 0:
    print("Global elapsed run time is: ", t_all, " seconds")


# finished, reconstruct the final solution for plotting
if rank == 0:
    sub_u = sub_u[:-1]
elif rank == size -1:
    sub_u = sub_u[1:]
else:
    sub_u = sub_u[1:-1]
comm.Gather(sub_u, u, root=0)
if rank ==0:
    matplotlib.use('Agg')
    fig = plt.figure()
    plt.contourf(X, Y, u.T)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig('{0}.png'.format(n))
    plt.close(fig)




