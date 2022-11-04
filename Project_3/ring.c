#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h>
#include <math.h>
// helper functions
float RandomNumber();
void print_matrix();
void print_float_vector();

int main(int argc, char *argv[]){
	// init MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Request request;
	MPI_Status status;
	
	// init matrices
	int N = pow(2, 3);  // matrix size NxN
    float matrixA[N][N];
	float matrixB[N][N];
	float matrixC[N][N];
	float partitionA[N*N]; // row partition of A in a vector
	float partitionB[N*N]; // column partition of B in a vector
	float partitionC[N*N];
	int localSize = N*N/size; // size of local blocks
	int localRCs = N/size; // #rows or columns in each processor
	float localA[localSize];
	float localB[localSize];
	float localC[localSize];
	int r, c;
	
	// population A B with random floats between -1 and 1
	if (rank == 0){
		srand(8); 
		for (r = 0; r < N; r++){
			for (c = 0; c < N; c++){
				matrixA[r][c] = RandomNumber();
				matrixB[r][c] = RandomNumber();
			}
		}
	}
	
	//////////////////// Start Cannon's Method /////////////////////////
	double T_total, T_total_global;
	T_total -= MPI_Wtime();
	// in Proc 0, make row/column partitions as a vector for scatter
	if (rank == 0){
		for (r =0; r < N; r++){
			for (c = 0; c < N; c++){
				partitionA[(r*N)+c] = matrixA[r][c];
			}
		}
		for (int c =0; c < N; c++){
			for (int r = 0; r < N; r++){
				partitionB[(c*N)+r] = matrixB[r][c];
			}
		}
	}
	MPI_Scatter(&partitionA, localSize, MPI_FLOAT, &localA, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&partitionB, localSize, MPI_FLOAT, &localB, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	int iter, e;
	int k = rank;  // keep track of where to put the sum
	float sum = 0;
	for (iter = 0; iter < size; iter++){ // loop through all blocks of rows
		for (r = 0; r < localRCs; r++){  //loop through local rows and columns to calc sum
			for (c = 0; c < localRCs;c++){
				for (e = 0; e < N;e++){ // loop through all elements in a row/column
					sum += localA[e + r*N]*localB[e + c*N];
				}
				localC[k *localRCs+c*N + r] = sum; //put the sum at the correct index
				sum = 0;
			}
		}
		if (k >= size -1)
			k = 0;
		else
			k++;
		// send local rows of matrix A to rank -1, i.e., roll up all rows of A by one processor unit
		MPI_Isend(&localA, localSize, MPI_FLOAT, (rank == 0 ? size-1 : rank -1), 0, MPI_COMM_WORLD, & request);
		MPI_Irecv(&localA, localSize, MPI_FLOAT, (rank == size-1 ? 0 : rank +1), 0, MPI_COMM_WORLD, & request);	
		// make sure all processors has received the information, then go to the next iteration
		MPI_Wait(&request, &status);
	}
	// gather all the info to the root proc 0.
	MPI_Gather(&localC, localSize, MPI_FLOAT, &partitionC, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (rank == 0){
		for (c = 0; c< N; c++){
			for (r = 0; r < N; r++){
				matrixC[r][c] = partitionC[(N*c) + r];
			}
		}
	}
	T_total += MPI_Wtime();
	MPI_Reduce(&T_total,&T_total_global, 1 ,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	if (rank == 0){
		printf("Total global elapsed running time for PMM is %f seconds \n", T_total_global);
	}
	
//////// For verification ////////////
	float matrixC_singleP[N][N];
	if (rank == 0){
		double T;
		T -= MPI_Wtime();
		for(r = 0; r < N; r++){
		    for(c = 0; c < N; c++){
		      matrixC_singleP[r][c] = 0;
		      for(k = 0; k < N; k++){
		        matrixC_singleP[r][c] += matrixA[r][k]*matrixB[k][c];
      			}
    		}
    	}
		printf("Matrix A is: \n");
		print_matrix(N, matrixA);
		printf("Matrix B is: \n");
		print_matrix(N, matrixB);
		printf("Matrix C is: \n");
		print_matrix(N, matrixC);
    	printf("Matrix C calculated using one processor is: \n");
		print_matrix(N, matrixC_singleP);
		T += MPI_Wtime();
		printf("Total elapsed running time for single processor MM is %f seconds \n", T);
  	}	
	MPI_Finalize();
}



float RandomNumber(){
    return -1+(2)*((float)rand())/(float)RAND_MAX;
}

void print_matrix(int N, float mat[N][N]){
	int i, j;
	printf("[");
	for(i=0;i<N-1;i++){
        for(j=0;j<N;j++){
            printf("%.6f ",mat[i][j]);
        }
        printf("\n");  
	}
	for(j=0;j<N;j++){
		printf("%.6f ",mat[N-1][j]);
	}
	printf("]\n");
}

void print_float_vector(float* v, int N){
	printf("[");
	int j;
	for (j=0; j<N; j++) {
	    printf("%.6f ", v[j]);
	}
    printf("]\n");
}

