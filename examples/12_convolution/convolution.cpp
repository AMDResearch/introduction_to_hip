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
Convolution GPU kernel
-------------------------------------------------- */
__global__ void gpu_convolution(double *d_input, double *d_filter, double *d_output, int N_input, int N_filter, int N_output, int filter_radius)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ( (col < (N_input - filter_radius) && row < (N_input - filter_radius)) && (col >= filter_radius && row >= filter_radius) ){

        int index_input  = N_input * row + col;
        int index_output = N_output * (row - filter_radius) + (col - filter_radius);
        double temp      = 0.0;

        for(int k=0; k<N_filter; k++){
            for(int l=0; l<N_filter; l++){

                int index_filter = N_filter * k + l;
                int index_subset = index_input + N_input * (k - filter_radius) + (l - filter_radius);
                
                    temp = temp + d_input[index_subset] * d_filter[index_filter];

                }
            }

            d_output[index_output] = temp;
        }
}

/* --------------------------------------------------
Convolution CPU kernel
-------------------------------------------------- */
void cpu_convolution(double *h_input, double *h_filter, double *h_output, int N_input, int N_filter, int N_output, int filter_radius)
{
    for(int i=filter_radius; i<N_input-filter_radius; i++){
        for(int j=filter_radius; j<N_input-filter_radius; j++){

            int index_input  = N_input * i + j;
            int index_output = N_output * (i - filter_radius) + (j - filter_radius);
            double temp      = 0.0;

            for(int k=0; k<N_filter; k++){
                for(int l=0; l<N_filter; l++){

                    int index_filter = N_filter * k + l;
                    int index_subset = index_input + N_input * (k - filter_radius) + (l - filter_radius);

                    temp = temp + h_input[index_subset] * h_filter[index_filter];
                }
            }

            h_output[index_output] = temp;
        }
    } 
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of square input, filter, and output matrices */
    int N_input  = 1024;
    int N_filter = 7;
    int N_output = N_input - N_filter + 1;

    int filter_radius = (N_filter - 1) / 2;

    /* Bytes in input, filter, and output matrices (double precision) */
    size_t bytes_input  = N_input * N_input * sizeof(double);
    size_t bytes_filter = N_filter * N_filter * sizeof(double);
    size_t bytes_output = N_output * N_output * sizeof(double);

    /* Allocate memory buffers for host input, filter, and output matrices */
    double *h_input           = (double*)malloc(bytes_input);
    double *h_filter          = (double*)malloc(bytes_filter);
    double *h_output          = (double*)malloc(bytes_output);
    double *h_output_from_gpu = (double*)malloc(bytes_output);

    /* Initialize host filter matrix */
    for(int i=0; i<N_filter*N_filter; i++){
        if( (i % 2) == 0 ){
            h_filter[i] = 0;
        }
        else{
            h_filter[i] = 1;
        }
    }

    /* Initialize host intput matrix */
    for(int i=0; i<N_input; i++){
        for(int j=0; j<N_input; j++){

            int index_input      = N_input * i + j;
            h_input[index_input] = (double)rand() / (double)RAND_MAX;

        }
    }   

    /* Initialize host output matrix */
    for(int i=0; i<N_output; i++){
        for(int j=0; j<N_output; j++){

            int index_output                = N_output * i + j;
            h_output[index_output]          = 0.0;
            h_output_from_gpu[index_output] = 0.0;

        }
    } 
   
    cpu_convolution(h_input, h_filter, h_output, N_input, N_filter, N_output, filter_radius);
 
    /* Allocate memory buffers for device input, filter, and output matrices */
    double *d_input, *d_filter, *d_output;
    gpuCheck( hipMalloc(&d_input, bytes_input) );
    gpuCheck( hipMalloc(&d_filter, bytes_filter) );
    gpuCheck( hipMalloc(&d_output, bytes_output) );

    /* Copy data from host matrices to device matrices */
    gpuCheck( hipMemcpy(d_input, h_input, bytes_input, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_filter, h_filter, bytes_filter, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_output, h_output, bytes_output, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid
    
       (NOTE: dim3 is a c struct with member variables x, y, z) */
    dim3 thr_per_blk( 16, 16, 1 );
    dim3 blk_in_grid( ceil( float(N_input) / thr_per_blk.x), ceil(float(N_input) / thr_per_blk.y), 1 );

    /* Launch matrix addition kernel */
    gpu_convolution<<<blk_in_grid, thr_per_blk>>>(d_input, d_filter, d_output, N_input, N_filter, N_output, filter_radius);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device matrix to host matrix (only need result, d_output) */
    gpuCheck( hipMemcpy(h_output_from_gpu, d_output, bytes_output, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double tolerance = 1.0e-14;

    for(int i=0; i<N_output; i++){
        for(int j=0; j<N_output; j++){

            int index_output = N_output * i + j;

            if( fabs(h_output_from_gpu[index_output] - h_output[index_output]) > tolerance ){
                printf("Error on element %d! Output from GPU: %0.14f, Output from CPU: %0.14f\n", index_output, h_output_from_gpu[index_output], h_output[index_output]);
                exit(1);
            }
            
        }
    } 

    /* Free CPU memory */
    free(h_input);
    free(h_filter);
    free(h_output);
    free(h_output_from_gpu);

    /* Free Device memory */
    gpuCheck( hipFree(d_input) );
    gpuCheck( hipFree(d_filter) );
    gpuCheck( hipFree(d_output) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N_input            : %d\n", N_input);
    printf("X Blocks in Grid   : %d\n", blk_in_grid.x);
    printf("X Threads per Block: %d\n", thr_per_blk.x);
    printf("Y Blocks in Grid   : %d\n", blk_in_grid.y);
    printf("Y Threads per Block: %d\n", thr_per_blk.y);
    printf("==============================\n\n");

    return 0;
}
