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
	int N = pow(2, 3);  // local array length, you can change this value if you like
	int local_array[N]; // init an array of all zeros, size N
	

	srand(rank+3);
    for (int i=0; i<N; i++) 
        local_array[i] = rand() % 100;
//	print_local_vector(local_array, N, rank);
	
	// Using my own function
	MY_Global_Max_Loc(local_array, N, MPI_INT, MPI_COMM_WORLD);
	
	// Using Reduce
	int localres[2];
    int globalres[2];
    int max_array[N];
	int maxloc_array[N];
    localres[1] = rank;
    for (int i=0; i<N; i++) {
	    localres[0] = local_array[i];
		MPI_Reduce(localres, globalres, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
		if(rank == 0){
		max_array[i] = globalres[0];
		maxloc_array[i] = globalres[1];
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		printf("The max_loc array given by MPI_reduce is: ");
		print_int_vector(maxloc_array, N);
	}
	MPI_Finalize();
}




void MY_Global_Max_Loc(int data[], int count, MPI_Datatype datatype, MPI_Comm communicator){
	int rank, size;
	MPI_Comm_rank(communicator, &rank);
	MPI_Comm_size(communicator, &size);
	int my_max[count];
	int my_maxloc[count];
	int temp[count];
	int temp_loc[count];
	int N = count;
	int j;
	int right = size -1;
	for (j=0; j<N; j++) {
		my_max[j] = data[j];
		my_maxloc[j] = rank;
	}
	
  	while(1){
  		if (right == 0){
  			break;
		}
		// bisection method to do inverse binomial tree
		int mid = (right +1)/2;
		if (right == 2 * mid){    // for the case when the size is an odd number
			if (rank > mid){      // second sub_network sends info to the first 
				MPI_Send(my_max, count, datatype, rank - mid, 0, communicator);
				MPI_Send(my_maxloc, count, datatype, rank - mid, 0, communicator);
			}
			if (rank <= mid && rank > 0){  // first sub_network receives info, compare, and update max loc array
				MPI_Recv(temp, count, datatype, rank + mid, 0, communicator, MPI_STATUS_IGNORE);	
				MPI_Recv(temp_loc, count, datatype, rank + mid, 0, communicator, MPI_STATUS_IGNORE);
					for (j=0; j<N; j++) {
						if (my_max[j] == temp[j]){
							if (temp_loc[j] < my_maxloc[j]){
								my_maxloc[j] = temp_loc[j];
							} 
						}
						if(my_max[j] < temp[j]){
							my_max[j] = temp[j];
							my_maxloc[j] = temp_loc[j];
						}
					}
			}
		}
		if (right != 2 * mid){   // for the case when the size is an even number
			if (rank >= mid){
				MPI_Send(my_max, count, datatype, rank - mid, 0, communicator);
				MPI_Send(my_maxloc, count, datatype, rank - mid, 0, communicator);
			}
			if (rank < mid){
				MPI_Recv(temp, count, datatype, rank + mid, 0, communicator, MPI_STATUS_IGNORE);	
				MPI_Recv(temp_loc, count, datatype, rank + mid, 0, communicator, MPI_STATUS_IGNORE);
					for (j=0; j<N; j++) {
						if (my_max[j] == temp[j]){
							if (temp_loc[j] < my_maxloc[j]){
								my_maxloc[j] = temp_loc[j];
							} 
						}
						if(my_max[j] < temp[j]){
							my_max[j] = temp[j];
							my_maxloc[j] = temp_loc[j];
						}
					}
			}
		}
		right = right/2;
		MPI_Barrier(communicator);
	}
	if(rank == 0){
		printf("MY max_loc array is: ");
		print_int_vector(my_maxloc, N);
	}

}







void print_local_vector(int* v, int N, int rank){
	printf("Local array from Processor %d: [", rank);
	int j;
	for (j=0; j<N; j++) {
	    printf("%d ", v[j]);
	}
    printf("]\n");
}


void print_int_vector(int* v, int N){
	printf("[");
	int j;
	for (j=0; j<N; j++) {
	    printf("%d ", v[j]);
	}
    printf("]\n");
}


