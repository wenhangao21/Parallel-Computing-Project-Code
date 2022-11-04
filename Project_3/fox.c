#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h>
#include <math.h>
// helper functions
float RandomNumber();
void print_matrix();
void print_float_vector();
int iter_diagonal();

// Uncomment Line 133 to display calculated matrix C, and you may verify with single-processor result.
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
	float localTempA[localSize];
	float localC[localSize];
	int r, c, i, j, k;
	
	// get partition parameters
	int XY = (int)sqrt(size);; // partitions over the x,y axes of the 3-D hypercube
	int xy = N/XY; // local blocks length and width
	int color = rank/XY;
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);
	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);
	
	// population A B with random floats between -1 and 1
	if (rank == 0){
		srand(7); 
		for (r = 0; r < N; r++){
			for (c = 0; c < N; c++){
				matrixA[r][c] = RandomNumber();
				matrixB[r][c] = RandomNumber();
			}
		}
	}
	if (rank == 0){
		for (r =0; r < XY; r++){
			for (c = 0; c < XY; c++){
				for (i = 0; i < xy; i++){
					for (j = 0; j < xy; j++){
						partitionA[localSize*(r*XY +c)+i*xy+j] = matrixA[r*xy + i][c*xy + j];
						partitionB[localSize*(r*XY +c)+i*xy+j] = matrixB[r*xy + i][c*xy + j];
					}
				}
			}
		}
	}
	///////////// Start Fox Method/////////////////
	double T_total, T_total_global;
	T_total -= MPI_Wtime();
	MPI_Scatter(&partitionA, localSize, MPI_FLOAT, &localTempA, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&partitionB, localSize, MPI_FLOAT, &localB, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);	
	int iter, e;
	float sum = 0;
	for (iter = 0; iter < XY; iter++){
		// if "diagonal" at current iter, localA is itself, and  send to everyone else
		int diagl = iter_diagonal(rank,XY, iter);
		if (rank == iter_diagonal(rank,XY, iter)){
			for (i = 0; i<localSize; i++){
				localA[i] = localTempA[i];
			}
			// send it to the whole row 
			}	
		MPI_Bcast(localA, localSize, MPI_FLOAT, diagl %XY, row_comm);
		// calc C
		for (r =0; r < xy; r++){	
				for (c = 0; c < xy; c++){
					for (e = 0; e < xy;e++){
						sum+=localA[e +r*xy]* localB[e*xy+c];
					}
					localC[r*xy+c] += sum;
					sum = 0;
				}
		}
		// shift B up by one block unit		
		MPI_Isend(&localB, localSize, MPI_FLOAT, (rank < XY ? rank + size-XY : rank -XY), 0, MPI_COMM_WORLD, & request);
		MPI_Irecv(&localB, localSize, MPI_FLOAT, (rank >= size-XY ? rank - size +XY : rank +XY), 0, MPI_COMM_WORLD, & request);	
		MPI_Wait(&request, &status);
	}
	MPI_Gather(&localC, localSize, MPI_FLOAT, &partitionC, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (rank == 0){
		for (r =0; r < XY; r++){
			for (c = 0; c < XY; c++){
				for (i = 0; i < xy; i++){
					for (j = 0; j < xy; j++){
						matrixC[r*xy + i][c*xy + j] = partitionC[localSize*(r*XY +c)+i*xy+j];
					}
				}
			}
		}
	}
	T_total += MPI_Wtime();
	MPI_Reduce(&T_total,&T_total_global, 1 ,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

	if (rank == 0){
		printf("Total global elapsed running time for PMM is %f seconds \n", T_total_global);
	}

////////// For verification ////////////
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
//		printf("Matrix A is: \n");
//		print_matrix(N, matrixA);
//		printf("Matrix B is: \n");
//		print_matrix(N, matrixB);
//		printf("Matrix C is: \n");
//		print_matrix(N, matrixC);
//    	printf("Matrix C calculated using one processor is: \n");
//		print_matrix(N, matrixC_singleP);
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

int iter_diagonal(int rank, int XY, int iter){
	int diag = rank/XY*XY + rank/XY;
	if (diag + iter < (rank/XY+1)*XY){
		return diag + iter;
	}
	else{
		return (diag + iter)-XY;
	}
}

