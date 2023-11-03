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
Matrix multiply kernel
-------------------------------------------------- */
__global__ void matrix_multiply(double *A, double *B, double *C, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < n && row < n){

        int index = n * row + col;
        double element = 0.0;

        for (int i=0; i<n; i++){

            int row_index = n * row + i;
            int col_index = n * i   + col;

            element = element + A[row_index] * B[col_index]; 
        }

        C[index] = element;
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of NxN matrix */
    int N = 1024;

    /* Bytes in matrix in double precision */
    size_t bytes = N * N * sizeof(double);

    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = N * i + j;

            h_A[index] = j + 1.0;
            h_B[index] = 1.0 / (i + 1.0);
            h_C[index] = 0.0;
        }
    }    

    /* Allocate memory for device matrices */
    double *d_A, *d_B, *d_C;
    gpuCheck( hipMalloc(&d_A, bytes) );
    gpuCheck( hipMalloc(&d_B, bytes) );
    gpuCheck( hipMalloc(&d_C, bytes) );

    /* Copy data from host matrices to device matricesL */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid
    
       (NOTE: dim3 is a c struct with member variables x, y, z) */
    dim3 thr_per_blk( 16, 16, 1 );
    dim3 blk_in_grid( ceil( float(N) / thr_per_blk.x), ceil(float(N) / thr_per_blk.y), 1 );

    /* Launch matrix addition kernel */
    matrix_multiply<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device matrix to host matrix (only need result, d_C) */
    gpuCheck( hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
                
            int index = N * i + j;
            if( fabs(h_C[index] - N ) > tolerance ){
                printf("Error: h_C[%d] = %0.14f instead of 1.00000000000000\n", index, h_C[index]);
                exit(1);
            }
        }
    }   

    /* Free CPU memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Free Device memory */
    gpuCheck( hipFree(d_A) );
    gpuCheck( hipFree(d_B) );
    gpuCheck( hipFree(d_C) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                  : %d\n", N);
    printf("X Blocks in Grid   : %d\n",  blk_in_grid.x);
    printf("X Threads per Block: %d\n",  thr_per_blk.x);
    printf("Y Blocks in Grid   : %d\n",  blk_in_grid.y);
    printf("Y Threads per Block: %d\n",  thr_per_blk.y);
    printf("==============================\n\n");

    return 0;
}