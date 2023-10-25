#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include <stdlib.h>

void PrintMatrix(double *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%.6e ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int main(int argc, char *argv[])
{
    int rank, num_procs;
    int size = 4; // 矩阵的大小,根据mpi_test_data.m的数据硬编码为指定大小
    double *A, *B, *C;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int block_size = size / (num_procs - 1);

    if (rank == 0)
    {
        // 初始化矩阵A、B、C
        // 注意：这里只有主进程需要分配内存
        double A[] = {
            // 填入mpi_test_data.m中的数据
            -1.000000e+00,
            -2.707955e-01,
            5.350056e-02,
            8.634630e-01,
            -9.980292e-01,
            -8.173388e-01,
            -9.113315e-02,
            1.361192e-01,
            -9.167380e-01,
            -8.154047e-01,
            -5.336431e-01,
            1.121887e-01,
            -6.467147e-01,
            -2.556555e-02,
            6.625836e-01,
            -8.983362e-01,

        };
        double B[] = {
            // 填入mpi_test_data.m中的数据
            5.341023e-01,
            7.519617e-01,
            6.208589e-01,
            -8.464509e-01,
            -9.621704e-01,
            6.311373e-02,
            -6.231595e-01,
            6.305478e-01,
            -4.952805e-01,
            8.405219e-01,
            7.726289e-01,
            9.697820e-01,
            -4.036057e-01,
            3.086230e-02,
            1.412280e-01,
            -7.632966e-01,
        };
        C = (double *)malloc(sizeof(double) * size * size);

        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                C[i * size + j] = 0.0;

        // 将A、B分块发送给其他进程
        for (int i = 1; i < num_procs; i++)
        {
            // 使用MPI_Send将A和B的块数据发送给其他进程
            MPI_Send(A + (i - 1) * block_size * size, block_size * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(B, size * size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }

        // 接收其他进程发送回来的结果C
        for (int i = 1; i < num_procs; i++)
        {
            // 使用MPI_Recv接收各个进程发送回来的结果
            MPI_Recv(C + (i - 1) * block_size * size, block_size * size, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 打印最终结果C
        PrintMatrix(C, size);

        // 释放内存
        free(C);
    }
    else
    {
        // 分配 partialA 和 partialC 的内存空间
        double *partialA = (double *)malloc(sizeof(double) * block_size * size);
        double *partialC = (double *)malloc(sizeof(double) * block_size * size);
        double *B = (double *)malloc(sizeof(double) * size * size);
        // 接收主进程发送过来的A和B
        MPI_Recv(partialA, block_size * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, size * size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 对接收的数据进行计算
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double temp = 0;
                for (int k = 0; k < size; k++)
                {
                    temp += partialA[i * size + k] * B[k * size + j];
                }
                partialC[i * size + j] = temp;
            }
        }

        // 将计算结果发送给主进程
        MPI_Send(partialC, block_size * size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

        // 释放内存
        free(partialA);
        free(B);
        free(partialC);
    }

    MPI_Finalize();
    return 0;
}