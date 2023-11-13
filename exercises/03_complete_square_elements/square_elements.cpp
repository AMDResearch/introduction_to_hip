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
Square elements kernel
-------------------------------------------------- */
__global__ void square_elements(int *A, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;


    // TODO: Complete the kernel by squaring
    //       all elements of the array.

}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of array */
    int N = 1024 * 1024;

    /* Bytes in N ints */
    size_t bytes = N * sizeof(int);

    /* Allocate memory for host array */
    int *h_A = (int*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        h_A[i] = i; 
    }    

    /* Allocate memory for device array */
    int *d_A;
    gpuCheck( hipMalloc(&d_A, bytes) );

    /* Copy data from host array to device array */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    /* Launch kernel */
    square_elements<<<blk_in_grid, thr_per_blk>>>(d_A, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device array to host array */
    gpuCheck( hipMemcpy(h_A, d_A, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    for (int i=0; i<N; i++){

        if(h_A[i] != i * i){
            printf("Error: h_A[%d] = %d instead of %d\n", i, h_A[i], i*i );
            exit(1);
        }
    }

    /* Free CPU memory */
    free(h_A);

    /* Free Device memory */
    gpuCheck( hipFree(d_A) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                : %d\n", N);
    printf("Blocks in Grid   : %d\n",  blk_in_grid);
    printf("Threads per Block: %d\n",  thr_per_blk);
    printf("==============================\n\n");

    return 0;
}
