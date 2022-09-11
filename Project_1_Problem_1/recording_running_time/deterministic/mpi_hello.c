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
	
	if(rank == 0){
		//Always print Processor 0 first
		printf("Hello from Processor %d\n",rank);
		
		// send message to Processor 1, where the message &rank is just a dummy variable that we send
		MPI_Send(&rank, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	}
	else{
		int dummy = 0; 
		// Processor k will wait to receive an int from Processor k -1 
		MPI_Recv(&dummy, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// Once received, print procid
		printf("Hello from Processor %d\n",rank);
		// Send info to its successor 
		if(rank <= size-2)
			MPI_Send(&dummy, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	time2 = MPI_Wtime();
	duration = time2 - time1;
	MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Global elapsed runtime is %f seconds \n", global);
	}
	MPI_Finalize();
}
