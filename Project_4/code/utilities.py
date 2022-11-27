import numpy as np


def jacobi(grid, f, dx, rank, size):
    newgrid = np.zeros(shape=grid.shape)
    # updating non-bdy points
    if rank == 0:
        newgrid[1:-1, 1:-1] = 0.25 * (grid[1:-1, :-2] + grid[1:-1, 2:] +
                                  grid[:-2, 1:-1] + grid[2:, 1:-1] - f[1:, 1:-1] * dx ** 2)
    elif rank == size -1:
        newgrid[1:-1, 1:-1] = 0.25 * (grid[1:-1, :-2] + grid[1:-1, 2:] +
                                  grid[:-2, 1:-1] + grid[2:, 1:-1] - f[0:-1, 1:-1] * dx ** 2)
    else:
        newgrid[1:-1, 1:-1] = 0.25 * (grid[1:-1, :-2] + grid[1:-1, 2:] +
                                  grid[:-2, 1:-1] + grid[2:, 1:-1] - f[:, 1:-1] * dx ** 2)

    # bdy points unchanged
    newgrid[0, :] = grid[0, :]
    newgrid[-1, :] = grid[-1, :]
    newgrid[:, 0] = grid[:, 0]
    newgrid[:, -1] = grid[:, -1]
    return newgrid


# initialize inner grid points with all 0s
def initgrid(gridsize):
    x = np.zeros((gridsize, gridsize))
    x[0, :] = 1
    x[:, -1] = 1
    x[-1, :] = 1
    x[:, 0] = 1

    return x


# exchange information, stripe partition
def exchange_info(rank, size, comm, sub_u):
    if rank == 0:
        comm.Send(sub_u[-2], dest=rank + 1)
        comm.Recv(sub_u[-1], source=rank + 1)
    elif rank == size - 1:
        comm.Send(sub_u[1], dest=rank - 1)
        comm.Recv(sub_u[0], source=rank - 1)
    else:
        comm.Send(sub_u[-2], dest=rank + 1)
        comm.Recv(sub_u[-1], source=rank + 1)
        comm.Send(sub_u[1], dest=rank - 1)
        comm.Recv(sub_u[0], source=rank - 1)
    comm.Barrier()
