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
Scale and shift kernel
-------------------------------------------------- */
__global__ void scale_and_shift(double *input, double *output, double scale, double shift, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) output[id] = scale * input[id] + shift;
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of array */
    int N = 1024 * 1024;

    /* Bytes in array in double precision */
    size_t bytes = N * sizeof(double);

    /* Allocate memory for host arrays */
    double *h_input  = (double*)malloc(bytes);
    double *h_output = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        h_input[i]  = (double)rand()/(double)RAND_MAX; 
        h_output[i] = 0.0;
    }    

    /* Allocate memory for device arrays */
    double *d_input, *d_output;
    gpuCheck( hipMalloc(&d_input, bytes) );
    gpuCheck( hipMalloc(&d_output, bytes) );

    /* Copy data from host arrays to device arrays */
    gpuCheck( hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_output, h_output, bytes, hipMemcpyHostToDevice) );

    /* Scale and shift values */
    double scale = 2.5;
    double shift = 1.2;

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    /* Launch vector addition kernel */
    scale_and_shift<<<blk_in_grid, thr_per_blk>>>(d_input, d_output, scale, shift, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device array to host array (only need result, d_C) */
    gpuCheck( hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double sum       = 0.0;
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        if (h_output[i] - (scale * h_input[i] + shift) > tolerance) {
            printf("Error: Sum/N = %0.2f instead of ~1.0\n", sum / N);
            exit(1);
        }
    }

    /* Free CPU memory */
    free(h_input);
    free(h_output);

    /* Free Device memory */
    gpuCheck( hipFree(d_input) );
    gpuCheck( hipFree(d_output) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                : %d\n", N);
    printf("Blocks in Grid   : %d\n",  blk_in_grid);
    printf("Threads per Block: %d\n",  thr_per_blk);
    printf("==============================\n\n");

    return 0;
}
