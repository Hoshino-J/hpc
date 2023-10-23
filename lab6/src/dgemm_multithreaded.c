#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MAX_SIZE 1000

double A[MAX_SIZE][MAX_SIZE];
double B[MAX_SIZE][MAX_SIZE];
double C[MAX_SIZE][MAX_SIZE];
int lda = MAX_SIZE;
int block_size = 64;
int num_threads = 12;

// Function for matrix multiplication (DGEMM)
void dgemm(int m, int n, int k, int beta, double A[][MAX_SIZE], double B[][MAX_SIZE], double C[][MAX_SIZE])
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = beta * C[i][j];
            for (int p = 0; p < k; p++)
            {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

// Thread function for parallel DGEMM
void *dgemm_thread(void *arg)
{
    long thread_id = (long)arg;
    int blocks_per_thread = lda / block_size / num_threads;
    int start_block = thread_id * blocks_per_thread;
    int end_block = (thread_id == num_threads - 1) ? (lda / block_size) : (start_block + blocks_per_thread);

    for (int i = start_block; i < end_block; i++)
    {
        for (int j = 0; j < lda; j += block_size)
        {
            for (int k = 0; k < lda; k += block_size)
            {
                dgemm(block_size, block_size, block_size, 1, A + i, B + k, C + i);
            }
        }
    }

    pthread_exit(NULL);
}

// Function to measure the performance and write GFLOPS to a file
void measure_performance(int matrix_size, double elapsed_time)
{
    double gflops = (2.0 * matrix_size * matrix_size * matrix_size) / (elapsed_time * 1e9);
    FILE *performance_file = fopen("performance.txt", "a");
    fprintf(performance_file, "%d\t%.2f\n", matrix_size, gflops);
    fclose(performance_file);
}

int main()
{
    // Initialize matrices A and B with random values
    for (int i = 0; i < MAX_SIZE; i++)
    {
        for (int j = 0; j < MAX_SIZE; j++)
        {
            A[i][j] = rand() / (double)RAND_MAX;
            B[i][j] = rand() / (double)RAND_MAX;
            C[i][j] = 0.0;
        }
    }

    // Measure performance and write to a file
    FILE *performance_file = fopen("performance.txt", "w");
    fprintf(performance_file, "Matrix Size\tGFLOPS\n");
    fclose(performance_file);

    for (int matrix_size = 64; matrix_size <= MAX_SIZE; matrix_size += 64)
    {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (matrix_size <= block_size)
        {
            // Use single-threaded DGEMM for small matrix sizes
            dgemm(matrix_size, matrix_size, matrix_size, 1, A, B, C);
        }
        else
        {
            // Use multi-threaded DGEMM for larger matrix sizes
            pthread_t threads[num_threads];

            for (long t = 0; t < num_threads; t++)
            {
                int rc = pthread_create(&threads[t], NULL, dgemm_thread, (void *)t);
                if (rc)
                {
                    printf("Error: unable to create thread, %d\n", rc);
                    exit(-1);
                }
            }

            for (long t = 0; t < num_threads; t++)
            {
                pthread_join(threads[t], NULL);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        measure_performance(matrix_size, elapsed_time);
    }

    return 0;
}