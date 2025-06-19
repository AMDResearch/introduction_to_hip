#include <stdio.h>
#include <math.h>
#include "hip/hip_runtime.h"

/* Macro for checking GPU API return values */
#define gpuCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

/* --------------------------------------------------
Matrix Transpose Kernel
-------------------------------------------------- */
__global__ void matrix_transpose(double* a, double* a_t, int m, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ( (col < n) && (row < m) ) {
        a_t[row * n + col] = a[col * m + row];
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of MxN array, where M is height and N is width */
    /*
    | A[0,0] ... A[0,N] |
    | .      .        . |
    | .       .       . |
    | .        .      . |
    | A[M,0] ... A[M,N] |
    */
    int M = 2048;
    int N = 1024;

    /* Bytes in array in double precision */
    size_t bytes = M * N * sizeof(double);

    /* Allocate memory for host arrays */
    double *h_A   = (double*)malloc(bytes);
    double *h_A_t = (double*)malloc(bytes);

    /* Initialize host arrays */
    for (int row=0; row<M; row++){
        for (int col=0; col<N; col++){
            h_A[row * N + col]   = (double)rand()/(double)RAND_MAX;
            h_A_t[row * N + col] = 0.0;
        }
    }

    /* Allocate memory for device arrays */
    double *d_A, *d_A_t;
    gpuCheck( hipMalloc(&d_A, bytes) );
    gpuCheck( hipMalloc(&d_A_t, bytes) );

    /* Copy data from host array to device array */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    dim3 thr_per_blk(16, 16, 1);
    dim3 blk_in_grid( ceil( float(N) / thr_per_blk.x ), ceil( float(M) / thr_per_blk.y ), 1 );

    /* Launch vector addition kernel */
    matrix_transpose<<<blk_in_grid, thr_per_blk>>>(d_A, d_A_t, M, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device array to host array (only need result, d_C) */
    gpuCheck( hipMemcpy(h_A_t, d_A_t, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    for (int row=0; row<M; row++){
        for (int col=0; col<N; col++){
            if (h_A_t[row * N + col] != h_A[col * M + row]){
                printf("Error: Mismatch at row = %d, column = %d\n", row, col);
                exit(1);
            }
        }
    }

    /* Free CPU memory */
    free(h_A);
    free(h_A_t);

    /* Free Device memory */
    gpuCheck( hipFree(d_A) );
    gpuCheck( hipFree(d_A_t) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("==============================\n\n");

    return 0;
}
