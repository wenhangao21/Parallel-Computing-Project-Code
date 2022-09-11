#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]){
	int rank, size;
	double time1, time2,duration,global;
	MPI_Init(&argc, &argv);
	// rank = processor number/id
	time1 = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// size = number of total processors
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	

	printf("Hello from Processor %d\n",rank);
	
	MPI_Barrier(MPI_COMM_WORLD);
	time2 = MPI_Wtime();
	duration = time2 - time1;
	MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Global elapsed runtime is %f seconds \n", global);
	}
	MPI_Finalize();
}
