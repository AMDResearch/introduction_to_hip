#include <stdio.h>
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

/* Number of elements in the array (without ghost zones) */
#define N (1024 * 1024 * 1024)

/* Stencil settings */
#define STENCIL_RADIUS 3
#define STENCIL_SIZE 7

/* Number of threads per block - defined here to use in kernel */
#define THREADS_PER_BLOCK 256

/* --------------------------------------------------
Stencil kernel
-------------------------------------------------- */
__global__ void stencil(double *a_in, double* a_out){

    /* Allocate shared memory */
    __shared__ double s_a_in[THREADS_PER_BLOCK + 2 * STENCIL_RADIUS];

    /* Define the global (id) and block-local (local_id) thread IDs offet by STENCIL_SIZE */
    int id       = blockDim.x * blockIdx.x + threadIdx.x + STENCIL_RADIUS;
    int local_id = threadIdx.x + STENCIL_RADIUS;

    /* Copy data from HBM buffer into shared-memory buffer: main array only, no ghost zones */
    s_a_in[local_id] = a_in[id];

    /* Copy data from HBM buffer into shared-memory buffer: leading/trailing ghost zones */
    if (threadIdx.x < STENCIL_RADIUS){
        s_a_in[local_id - STENCIL_RADIUS] = a_in[id - STENCIL_RADIUS];
        s_a_in[local_id + THREADS_PER_BLOCK] = a_in[id + THREADS_PER_BLOCK];
    }

    /* Ensure all threads in a block have finished copying their data into shared memory */
     __syncthreads();

    /* Perform average stencil operation for main elements of the array - not on ghost zones*/
    if (id < (N + STENCIL_RADIUS)){
        double sum = 0.0;

        for(int i=-STENCIL_RADIUS; i<=STENCIL_RADIUS; i++){
            sum += s_a_in[local_id + i];
        }

        a_out[id] = sum / STENCIL_SIZE;
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Number of bytes in array (including ghost zones) */
    size_t bytes = (N + 2 * STENCIL_RADIUS) * sizeof(double);

    /* Allocate host buffers */
    double *h_A_in  = (double*)malloc(bytes);
    double *h_A_out = (double*)malloc(bytes);

    /* Allocate device buffers */
    double *d_A_in, *d_A_out;
    gpuCheck( hipMalloc(&d_A_in, bytes) );
    gpuCheck( hipMalloc(&d_A_out, bytes) );

    /* Initialize array h_A_in with 2 for even and 1 for odd, and h_A_out with 0 */
    for(int i=0; i<(N+2*STENCIL_RADIUS); i++){

        if(i % 2 == 0){
            h_A_in[i] = 2.0;
        }
        else{
            h_A_in[i] = 1.0;
        }

        h_A_out[i] = 0.0;
    }

    /* Copy arrays from host to device */
    gpuCheck( hipMemcpy(d_A_in, h_A_in, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_A_out, h_A_out, bytes, hipMemcpyHostToDevice) );

    /* Number of thread blocks in grid */
    int blk_in_grid = ceil( float(N) / THREADS_PER_BLOCK );

    /* Launch stencil kernel */
    stencil<<<blk_in_grid, THREADS_PER_BLOCK>>>(d_A_in, d_A_out);

    /* Copy out-array from device to host */
    gpuCheck( hipMemcpy(h_A_out, d_A_out, bytes, hipMemcpyDeviceToHost) );

    /* Check results */
    double tolerance   = 1.0e-14;
    double odd_result  = (2.0 + 1.0 + 2.0 + 1.0 + 2.0 + 1.0 + 2.0) / (double)STENCIL_SIZE;
    double even_result = (1.0 + 2.0 + 1.0 + 2.0 + 1.0 + 2.0 + 1.0) / (double)STENCIL_SIZE;

    for(int i=STENCIL_RADIUS; i<(STENCIL_RADIUS+N); i++){

        if(i % 2 == 0){

            if( fabs( (h_A_out[i] - even_result) ) > tolerance ){
                printf("Error: Even result is %0.14f instead of %0.14f (element %d). Exiting...\n", 
                       h_A_out[i], even_result, i);

                exit(1);
            }
        }
        else{

            if( fabs( (h_A_out[i] - odd_result) ) > tolerance ){
                printf("Error: Odd result is %0.14f instead of %0.14f (element %d). Exiting...\n",
                       h_A_out[i], odd_result, i);

                exit(1);
            }
        }
    }

    /* Free host memory */
    free(h_A_in);
    free(h_A_out);

    /* Free device memory */
    gpuCheck( hipFree(d_A_in) );
    gpuCheck( hipFree(d_A_out) );

    printf("__SUCCESS__\n");

    return 0;
}
