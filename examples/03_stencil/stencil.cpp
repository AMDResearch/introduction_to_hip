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

/* Stencil kernel */
__global__ void stencil(double *a_in, double* a_out, int n, int radius, int size){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    /* If we're not in the ghost zones... */    
    if( (id >= radius) && (id < (n + radius)) ){

        /* Sum the elements in the stencil for id element */ 
        double temp = 0.0;
        for(int i=-radius; i<=radius; i++){
            temp += a_in[id + i];
        }

        /* Divide by stencil size to get average */
        a_out[id] = temp / (double)size;
    }
}

/* Main program */
int main(int argc, char *argv[]){

    /* Number of elements in array (excluding ghost zones) */
    int N = 1024 * 1024;

    /* Stencil size */
    int stencil_size = 7;

    /* Stencil radius */
    int stencil_radius = 3;

    /* Number of bytes in array (including ghost zones) */
    size_t bytes = (N + 2 * stencil_radius) * sizeof(double);

    /* Allocate host buffers */
    double *h_A_in  = (double*)malloc(bytes);
    double *h_A_out = (double*)malloc(bytes);

    /* Allocate device buffers */
    double *d_A_in, *d_A_out;
    gpuCheck( hipMalloc(&d_A_in, bytes) );
    gpuCheck( hipMalloc(&d_A_out, bytes) );

    /* Initialize array h_A_in with 2 for even and 1 for odd, and h_A_out with 0 */
    for(int i=0; i<(N+2*stencil_radius); i++){

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

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N + 2 * stencil_radius) / thr_per_blk );

    /* Launch stencil kernel */
    stencil<<<blk_in_grid, thr_per_blk>>>(d_A_in, d_A_out, N, stencil_radius, stencil_size);

    /* Copy out-array from device to host */
    gpuCheck( hipMemcpy(h_A_out, d_A_out, bytes, hipMemcpyDeviceToHost) );

    /* Check results */
    double tolerance   = 1.0e-14;
    double odd_result  = (2.0 + 1.0 + 2.0 + 1.0 + 2.0 + 1.0 + 2.0) / (double)stencil_size;
    double even_result = (1.0 + 2.0 + 1.0 + 2.0 + 1.0 + 2.0 + 1.0) / (double)stencil_size;

    for(int i=stencil_radius; i<(stencil_radius+N); i++){

        if(i % 2 == 0){

            if( fabs( (h_A_out[i] - even_result) ) > tolerance ){
                printf("Error: Even result is %0.14f instead of %0.14f. Exiting...\n", h_A_out[i], even_result);
                exit(1);
            }
        }
        else{

            if( fabs( (h_A_out[i] - odd_result) ) > tolerance ){
                printf("Error: Odd result is %0.14f instead of %0.14f. Exiting...\n", h_A_out[i], odd_result);
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
