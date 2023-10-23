# code:utf-8

import random
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt   


TOLERANCE = 1e-10
n_start = 40
n_end = 600 
n_inc = 80

result = []

def dgemm_compare(n):
    #populate the matrices with random values between 0.0 and 1.0
    A = [[random.random() for row in range(n)] for col in range(n)]
    B = [[random.random() for row in range(n)] for col in range(n)]
    C = [[0 for row in range(n)] for col in range(n)]
    C_np = [[0 for row in range(n)] for col in range(n)]
    # https://realpython.com/python-data-types/
    # https://stackoverflow.com/questions/6663272/double-precision-floating-values-in-python
    # https://docs.python.org/3/tutorial/floatingpoint.html

    nrepeats = 1  # 3
    gflops = n*n/(100*1000*1000) *2*n
    elapsed_time = 1000000000
    for i in range(0,nrepeats):
        C = [[0 for row in range(n)] for col in range(n)]  # np.dot没有累加，每次计算前清零
        print(time.time())
        start = time.time()
        #matrix multiplication
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]

        end = time.time()
        print(time.time())
        tmp_time = end - start
        if elapsed_time > tmp_time:
            elapsed_time = tmp_time

    np_elapsed_time = 1000000000
    for i in range(0,nrepeats):
        start = time.time()
        C_np = np.dot(A, B)
        end  = time.time()
        tmp_time = end - start
        if np_elapsed_time > tmp_time:
            np_elapsed_time = tmp_time

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if(abs(C[i][j] - C_np[i][j]) > TOLERANCE):
                    print("C[%d][%d] != C_np\n, %lf, %lf\n" %(i,j, C[i][j], C_np[i][j]))

    row = [n,n,n, gflops/elapsed_time, gflops/np_elapsed_time]
    print("%5d\t %5d\t %5d\t %5.2lf\t %5.2lf" % (n,n,n, gflops/elapsed_time, gflops/np_elapsed_time))

    return row


def load_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            if not line or line.startswith('%'): 
                continue

            values = [float(s) for s in line.split()]
            data.append(values)
    return data


def plot(data):
    data = np.array(data)
    x = data[:, 0]        # 切片取第1列
    y_py = data[:,-2]     # 取倒数第2列
    y_np = data[:,-1]     # 取倒数第1列

    plt.plot(x,y_np,color='g', linewidth=3, linestyle='-',label='numpy dgemm')
    plt.plot(x,y_py,color='c', linewidth=2, linestyle='--',label='python dgemm')
    
    plt.title('gemm optimize VS OpenBlas')
    plt.xlabel('M=N=K')
    plt.ylabel('gflops')
    plt.legend()

    plt.savefig('python_dgemm.jpg')

    plt.show()


print("%m\t%n\t%k\t%py_GFLOPS\t%np_GFLOPS")
for n in range(n_start, n_end, n_inc):
    row = dgemm_compare(n)
    result.append(row)

# data = load_data('python.m')
plot(result)
