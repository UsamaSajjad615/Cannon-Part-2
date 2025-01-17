#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
// This is also working perfectly but not fully follow task b only one point checkboard dsitrubutin miss,new task B fully follow and is more manuall approach.
#define MATRIX_SIZE 4  // Change this to test different matrix sizes

// Function to fill a matrix with random integer values
void fillMatrix(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 10 + 1;
    }
}

// Function to print a matrix
void printMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Cannon's algorithm for matrix multiplication
void cannonAlgorithm(int* localA, int* localB, int* localC, int blockSize, int rank, int dims[2]) {
    int left, right, up, down;

    int row = rank / dims[1];
    int col = rank % dims[1];

    // Calculate ranks of neighbors
    left = (col == 0) ? rank + dims[1] - 1 : rank - 1;
    right = (col == dims[1] - 1) ? rank - dims[1] + 1 : rank + 1;
    up = (row == 0) ? rank + (dims[0] - 1) * dims[1] : rank - dims[1];
    down = (row == dims[0] - 1) ? rank - (dims[0] - 1) * dims[1] : rank + dims[1];

    // Initial skewing
    for (int i = 0; i < row; i++) {
        MPI_Sendrecv_replace(localA, blockSize*blockSize, MPI_INT, left, 0, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < col; i++) {
        MPI_Sendrecv_replace(localB, blockSize*blockSize, MPI_INT, up, 0, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Main computation loop
    for (int step = 0; step < dims[0]; step++) {
        // Local matrix multiplication
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int k = 0; k < blockSize; k++) {
                    localC[i*blockSize + j] += localA[i*blockSize + k] * localB[k*blockSize + j];
                }
            }
        }

        // Shift matrices
        MPI_Sendrecv_replace(localA, blockSize*blockSize, MPI_INT, left, 0, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB, blockSize*blockSize, MPI_INT, up, 0, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = MATRIX_SIZE;
    
    // Create processor grid
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    
    if (dims[0] != dims[1] || dims[0]*dims[1] != size || n % dims[0] != 0) {
        if (rank == 0) {
            printf("Invalid processor grid! Requirements:\n");
            printf("1. Square grid (dims[0] = dims[1] = %d)\n", dims[0]);
            printf("2. n mod dims[0] = 0 (n=%d, dims[0]=%d)\n", n, dims[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int blockSize = n / dims[0];
    int *matrixA = NULL, *matrixB = NULL, *matrixC = NULL;
    int *localA = malloc(blockSize * blockSize * sizeof(int));
    int *localB = malloc(blockSize * blockSize * sizeof(int));
    int *localC = calloc(blockSize * blockSize, sizeof(int));

    // Initialize matrices on root
    if (rank == 0) {
        matrixA = malloc(n * n * sizeof(int));
        matrixB = malloc(n * n * sizeof(int));
        matrixC = calloc(n * n, sizeof(int));
        
        srand(time(NULL));
        fillMatrix(matrixA, n*n);
        fillMatrix(matrixB, n*n);
        
        printf("Matrix A:\n");
        printMatrix(matrixA, n, n);
        printf("Matrix B:\n");
        printMatrix(matrixB, n, n);
    }

    // Create block datatype
    MPI_Datatype blockType, resizedBlockType;
    MPI_Type_vector(blockSize, blockSize, n, MPI_INT, &blockType);
    MPI_Type_create_resized(blockType, 0, blockSize * sizeof(int), &resizedBlockType);
    MPI_Type_commit(&resizedBlockType);

    // Calculate displacements for checkerboard distribution
    int* sendCounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[1]; j++) {
                sendCounts[i*dims[1]+j] = 1;
                displs[i*dims[1]+j] = i * n + j;  
            }
        }
    }

    // Scatter matrices
    MPI_Scatterv(matrixA, sendCounts, displs, resizedBlockType,
                localA, blockSize*blockSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(matrixB, sendCounts, displs, resizedBlockType,
                localB, blockSize*blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Run Cannon's algorithm
    cannonAlgorithm(localA, localB, localC, blockSize, rank, dims);

    // Gather results
    MPI_Gatherv(localC, blockSize*blockSize, MPI_INT,
               matrixC, sendCounts, displs, resizedBlockType, 0, MPI_COMM_WORLD);

    // Print results
    if (rank == 0) {
        printf("Matrix C (Result):\n");
        printMatrix(matrixC, n, n);
        free(matrixA);
        free(matrixB);
        free(matrixC);
    }

    // Cleanup
    free(localA);
    free(localB);
    free(localC);
    free(sendCounts);
    free(displs);
    MPI_Type_free(&resizedBlockType);
    MPI_Finalize();
    return 0;
}// Commit for 2025-01-17 12:00:00 - Added print function for matrices
