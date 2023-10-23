#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h> // Include OpenBLAS header

#define M 3
#define N 3
#define K 3

// Function to generate random double precision values between 0 and 1
double random_double()
{
    return (double)rand() / RAND_MAX;
}

void initialize_matrix(int row, int col, double matrix[row][col])
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            matrix[i][j] = random_double();
        }
    }
}

void print_matrix(int row, int col, double matrix[row][col])
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    srand(time(NULL)); // Seed the random number generator with current time

    double A[M][K];
    double B[K][N];
    double C[M][N];

    // Initialize matrices A and B with random values
    initialize_matrix(M, K, A);
    initialize_matrix(K, N, B);

    // Perform matrix multiplication C = A * B using OpenBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, (double *)A, K, (double *)B, N, 0.0, (double *)C, N);

    printf("Matrix A:\n");
    print_matrix(M, K, A);

    printf("Matrix B:\n");
    print_matrix(K, N, B);

    printf("Matrix C (Result of A * B):\n");
    print_matrix(M, N, C);

    return 0;
}