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

#define N (1024 * 1024)
#define THREADS_PER_BLOCK 256

/* --------------------------------------------------
Vector addition kernel
-------------------------------------------------- */
__global__ void vector_addition(double *A, double *B, double *C)
{
    __shared__ double s_A[THREADS_PER_BLOCK];
    __shared__ double s_B[THREADS_PER_BLOCK];

    int id  = blockDim.x * blockIdx.x + threadIdx.x;
    int lid = threadIdx.x;

    s_A[lid] = A[id];
    s_B[lid] = B[id];

    __syncthreads();

    if (id < N){
        double temp = s_A[lid] + s_B[lid];
        C[id] = temp;
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Bytes in array in double precision */
    size_t bytes = N * sizeof(double);

    /* Allocate memory for host arrays */
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        h_A[i] = sin(i) * sin(i); 
        h_B[i] = cos(i) * cos(i);
        h_C[i] = 0.0;
    }    

    /* Allocate memory for device arrays */
    double *d_A, *d_B, *d_C;
    gpuCheck( hipMalloc(&d_A, bytes) );
    gpuCheck( hipMalloc(&d_B, bytes) );
    gpuCheck( hipMalloc(&d_C, bytes) );

    /* Copy data from host arrays to device arrays */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice) );

    /*  blk_in_grid: number of thread blocks in grid */
    int blk_in_grid = ceil( float(N) / THREADS_PER_BLOCK );

    /* Launch vector addition kernel */
    vector_addition<<<blk_in_grid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device array to host array (only need result, d_C) */
    gpuCheck( hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double sum       = 0.0;
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        sum = sum + h_C[i];
    } 

    if( fabs( (sum / N) - 1.0 ) > tolerance ){
        printf("Error: Sum/N = %0.2f instead of ~1.0\n", sum / N);
        exit(1);
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
    printf("N                : %d\n", N);
    printf("Blocks in Grid   : %d\n",  blk_in_grid);
    printf("Threads per Block: %d\n",  THREADS_PER_BLOCK);
    printf("==============================\n\n");

    return 0;
}
