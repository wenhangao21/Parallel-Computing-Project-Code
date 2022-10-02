#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]){
	int rank, size;
	MPI_Init(&argc, &argv);
	// rank = processor number/id
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// size = number of total processors
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// init variables for performance measures
	double time_MY_bcast_oneToAll = 0;
	double time_MPI_bcast = 0;
	double time_MY_bcast_binomialTree = 0;
	double global_MY_bcast_oneToAll, global_MPI_bcast, global_MY_bcast_binomialTree;
	///////////////////// Change Data Size here ////////////////////////////////
	int N = pow(2, 14);  // data size
	float data[N]; // init an array of all zeros, size N
	int trials = 10;
	
	// record time MY_bcast_oneToAll
	int i;
	
	// let 10 times to reduce timre measurement anomalies
	for (i = 0; i < trials; i++) {
		if (rank == 0) {
			data[1] = 1.0;
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		time_MY_bcast_oneToAll -= MPI_Wtime();
		MY_bcast_oneToAll(&data, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		time_MY_bcast_oneToAll += MPI_Wtime();
		

		// record time MY_bcast_binomialTree
		if (rank == 0) {
			data[3] = 3.0;
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		time_MY_bcast_binomialTree -= MPI_Wtime();
		MY_bcast_binomialTree(&data, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		time_MY_bcast_binomialTree += MPI_Wtime();
		
		// record time MPI_bcast
		if (rank == 0) {
			data[2] = 2.0;
		}	
		MPI_Barrier(MPI_COMM_WORLD);
		time_MPI_bcast -= MPI_Wtime();
		MPI_Bcast(&data, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		time_MPI_bcast += MPI_Wtime();
	}
  	// record max time as global elapsed runtime
	MPI_Reduce(&time_MY_bcast_oneToAll,&global_MY_bcast_oneToAll, 1 ,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&time_MPI_bcast,&global_MPI_bcast, 1 ,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&time_MY_bcast_binomialTree,&global_MY_bcast_binomialTree, 1 ,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Performance with %d processors and a data-size of %d \n", size, N);
		printf("Average global elapsed runtime for MY_bcast_oneToall is %f seconds \n", global_MY_bcast_oneToAll/trials);
		printf("Average global elapsed runtime for MPI_Bcast is %f seconds \n", global_MPI_bcast/trials);
		printf("Average global elapsed runtime for MY_bcast_binomialTree is %f seconds \n", global_MY_bcast_binomialTree/trials);
	}
	// uncomment to check if broadcast is done properly 
//	if(rank == size-1) {
//	printf("the first element is %f, the second element is %f, the third element is %f, the 4th element is %f\n", data[0], data[1], data[2], data[3]);
//	}
	MPI_Finalize();
}


void MY_bcast_oneToAll(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator) {
	int rank, size;
	MPI_Comm_rank(communicator, &rank);
	MPI_Comm_size(communicator, &size);

  	if (rank == root) {
	// send data from root to all other processors
	    int i;
	    for (i = 0; i < size; i++) {
	      if (i != rank) {
	        MPI_Send(data, count, datatype, i, 0, communicator);
	      }
	    }
    } else {
    // receive data from the root
    MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
	}
}
	

void MY_bcast_binomialTree(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator) {
	int rank, size;
	MPI_Comm_rank(communicator, &rank);
	MPI_Comm_size(communicator, &size);
	// broadcast through binomial tree 
	// implemented by bisection method
	if (root != 0){
		if (rank == root){
			MPI_Send(data, count, datatype, 0, 0, communicator);
		}
		if (rank == 0){
				MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
			}
	}
	int left = 0;
	int right = size -1;
  	while(1){
  		if (left == right){
  			break;
		}
		int mid = (left + right +1)/2;
		if (rank >= left && rank < mid){
			if (rank == left){
				MPI_Send(data, count, datatype, mid, 0, communicator);
			}
			right = mid -1;
		}
		else{
			if (rank == mid){
				MPI_Recv(data, count, datatype, left, 0, communicator, MPI_STATUS_IGNORE);
			}
			left = mid;
	
	  }
	}
}
