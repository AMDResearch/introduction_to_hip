#include <stdio.h>
#include "hip/hip_runtime.h"
#include <sys/time.h>

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
#define N (1024 * 1024 * 256)

/* Stencil settings */
#define STENCIL_RADIUS 3
#define STENCIL_SIZE 7

/* Number of threads per block - defined here to use in kernel */
#define THREADS_PER_BLOCK 256

/* --------------------------------------------------
Stencil kernel
-------------------------------------------------- */
__global__ void stencil(double *a_in, double *a_out){

    /* Define the global thread ID */
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N){

        double sum = 0.0;
   
        /* Perform stencil operation */ 
        for (int j=-STENCIL_RADIUS; j<=STENCIL_RADIUS; j++){

            if ( ((id+j) >= 0) && ((id+j) < N) ){
                sum = sum + a_in[id + j];
            }
        }
    
        a_out[id] = sum / (double)STENCIL_SIZE;
    }
}

/* --------------------------------------------------
CPU stencil
-------------------------------------------------- */
void cpu_stencil(double *in, double* out){

    for (int i=0; i<N; i++){

        double sum = 0.0;

        for (int j=-STENCIL_RADIUS; j<=STENCIL_RADIUS; j++){
  
            if ( ((i+j) >=0) && ((i+j) < N) ){
                sum = sum + in[i+j];
            }
        }
        
        out[i] = sum / (double)STENCIL_SIZE;
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Number of bytes in array */
    size_t bytes = N * sizeof(double);

    /* Allocate host buffers */
    double *h_A_in, *h_A_out, *h_A_out_cpu;
    gpuCheck( hipHostMalloc(&h_A_in, bytes) );
    gpuCheck( hipHostMalloc(&h_A_out, bytes) );
    gpuCheck( hipHostMalloc(&h_A_out_cpu, bytes) );

    /* Initialize CPU arrays */
    for (int i=0; i<N; i++){

        h_A_in[i]      = (double)rand()/(double)RAND_MAX;
        h_A_out[i]     = 0.0;
        h_A_out_cpu[i] = 0.0;
    }

    /* Allocate device buffers */
    double *d_A_in, *d_A_out;
    gpuCheck( hipMalloc(&d_A_in, bytes) );
    gpuCheck( hipMalloc(&d_A_out, bytes) );

    /* Number of thread blocks in grid */
    int blk_in_grid = ceil( float(N) / THREADS_PER_BLOCK );

    /* Copy arrays from host to device */
    gpuCheck( hipMemcpyAsync(d_A_in, h_A_in, bytes, hipMemcpyHostToDevice, NULL) );
    gpuCheck( hipMemcpyAsync(d_A_out, h_A_out, bytes, hipMemcpyHostToDevice, NULL) );

    /* Launch stencil kernel */
    stencil<<<blk_in_grid, THREADS_PER_BLOCK>>>(d_A_in, d_A_out);

    /* Copy out-array from device to host */
    gpuCheck( hipMemcpyAsync(h_A_out, d_A_out, bytes, hipMemcpyDeviceToHost, NULL) );

    /* Run CPU-version of stencil */
    cpu_stencil(h_A_in, h_A_out_cpu);

    gpuCheck( hipDeviceSynchronize() );

    /* Check results */
    double tolerance   = 1.0e-14;
    for (int i=0; i<N; i++){

        if ( fabs(h_A_out[i] - h_A_out_cpu[i]) > tolerance){
            printf("Error: h_A_out[%d] - h_A_out_cpu[%d] = %0.14f - %0.14f = %0.14f > %0.14f\n", i, i, h_A_out[i], h_A_out_cpu[i], fabs(h_A_out[i] - h_A_out_cpu[i]), tolerance);
            exit(1);
        }
    }

    /* Free host memory */
    gpuCheck( hipHostFree(h_A_in) );
    gpuCheck( hipHostFree(h_A_out) );
    gpuCheck( hipHostFree(h_A_out_cpu) );

    /* Free device memory */
    gpuCheck( hipFree(d_A_in) );
    gpuCheck( hipFree(d_A_out) );

    printf("__SUCCESS__\n");

    return 0;
}
