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
Matrix multiply kernel 1
-------------------------------------------------- */
__global__ void matrix_multiply_1(double *A, double *B, double *C, int n)
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
Matrix multiply kernel 2
-------------------------------------------------- */
__global__ void matrix_multiply_2(double *A, double *B, double *C, int n)
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
Matrix multiply kernel 3
-------------------------------------------------- */
__global__ void matrix_multiply_3(double *A, double *B, double *C, int n)
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

    /* Allocate memory for host matrices 1 */
    double *h_A_1 = (double*)malloc(bytes);
    double *h_B_1 = (double*)malloc(bytes);
    double *h_C_1 = (double*)malloc(bytes);

    /* Allocate memory for host matrices 2 */
    double *h_A_2 = (double*)malloc(bytes);
    double *h_B_2 = (double*)malloc(bytes);
    double *h_C_2 = (double*)malloc(bytes);

    /* Allocate memory for host matrices 3 */
    double *h_A_3 = (double*)malloc(bytes);
    double *h_B_3 = (double*)malloc(bytes);
    double *h_C_3 = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = N * i + j;

            h_A_1[index] = j + 1.0;
            h_B_1[index] = 1.0 / (i + 1.0);
            h_C_1[index] = 0.0;

            h_A_2[index] = j + 1.0;
            h_B_2[index] = 1.0 / (i + 1.0);
            h_C_2[index] = 0.0;

            h_A_3[index] = j + 1.0;
            h_B_3[index] = 1.0 / (i + 1.0);
            h_C_3[index] = 0.0;

        }
    }    

    hipStream_t stream1;
    gpuCheck( hipStreamCreate(&stream1) );

    hipStream_t stream2;
    gpuCheck( hipStreamCreate(&stream2) );

    hipStream_t stream3;
    gpuCheck( hipStreamCreate(&stream3) );

    /* Allocate memory for device matrices 1 */
    double *d_A_1, *d_B_1, *d_C_1;
    gpuCheck( hipMalloc(&d_A_1, bytes) );
    gpuCheck( hipMalloc(&d_B_1, bytes) );
    gpuCheck( hipMalloc(&d_C_1, bytes) );

    /* Allocate memory for device matrices 2 */
    double *d_A_2, *d_B_2, *d_C_2;
    gpuCheck( hipMalloc(&d_A_2, bytes) );
    gpuCheck( hipMalloc(&d_B_2, bytes) );
    gpuCheck( hipMalloc(&d_C_2, bytes) );

    /* Allocate memory for device matrices 3 */
    double *d_A_3, *d_B_3, *d_C_3;
    gpuCheck( hipMalloc(&d_A_3, bytes) );
    gpuCheck( hipMalloc(&d_B_3, bytes) );
    gpuCheck( hipMalloc(&d_C_3, bytes) );

    /* Copy data from host matrices 1 to device matrices 1 */
    gpuCheck( hipMemcpy(d_A_1, h_A_1, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B_1, h_B_1, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C_1, h_C_1, bytes, hipMemcpyHostToDevice) );

    /* Copy data from host matrices 2 to device matrices 2 */
    gpuCheck( hipMemcpy(d_A_2, h_A_2, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B_2, h_B_2, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C_2, h_C_2, bytes, hipMemcpyHostToDevice) );

    /* Copy data from host matrices 3 to device matrices 3 */
    gpuCheck( hipMemcpy(d_A_3, h_A_3, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B_3, h_B_3, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C_3, h_C_3, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid
    
       (NOTE: dim3 is a c struct with member variables x, y, z) */
    dim3 thr_per_blk( 16, 16, 1 );
    dim3 blk_in_grid( ceil( float(N) / thr_per_blk.x), ceil(float(N) / thr_per_blk.y), 1 );

    /* Launch matrix addition kernel 1 */
    matrix_multiply_1<<<blk_in_grid, thr_per_blk, 0, stream1>>>(d_A_1, d_B_1, d_C_1, N);

    /* Launch matrix addition kernel 2 */
    matrix_multiply_2<<<blk_in_grid, thr_per_blk, 0, stream2>>>(d_A_2, d_B_2, d_C_2, N);

    /* Launch matrix addition kernel 3 */
    matrix_multiply_3<<<blk_in_grid, thr_per_blk, 0, stream3>>>(d_A_3, d_B_3, d_C_3, N);

    /* Copy data from device matrix to host matrix 1 (only need result, d_C_1) */
    gpuCheck( hipMemcpy(h_C_1, d_C_1, bytes, hipMemcpyDeviceToHost) );

    /* Copy data from device matrix to host matrix 2 (only need result, d_C_2) */
    gpuCheck( hipMemcpy(h_C_2, d_C_2, bytes, hipMemcpyDeviceToHost) );

    /* Copy data from device matrix to host matrix 3 (only need result, d_C_3) */
    gpuCheck( hipMemcpy(h_C_3, d_C_3, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
                
            int index = N * i + j;

            if( isnan(h_C_1[index]) || fabs(h_C_1[index] - N ) > tolerance ){
                printf("Error: h_C_1[%d] = %0.14f instead of %d\n", index, h_C_1[index], N);
                exit(1);
            }

            if( isnan(h_C_2[index]) || fabs(h_C_2[index] - N ) > tolerance ){
                printf("Error: h_C_2[%d] = %0.14f instead of %d\n", index, h_C_2[index], N);
                exit(1);
            }

            if( isnan(h_C_3[index]) || fabs(h_C_3[index] - N ) > tolerance ){
                printf("Error: h_C_3[%d] = %0.14f instead of %d\n", index, h_C_3[index], N);
                exit(1);
            }
        }
    }   

    /* Free CPU memory 1 */
    free(h_A_1);
    free(h_B_1);
    free(h_C_1);

    /* Free CPU memory 2 */
    free(h_A_2);
    free(h_B_2);
    free(h_C_2);

    /* Free CPU memory 3 */
    free(h_A_3);
    free(h_B_3);
    free(h_C_3);

    /* Free Device memory 1 */
    gpuCheck( hipFree(d_A_1) );
    gpuCheck( hipFree(d_B_1) );
    gpuCheck( hipFree(d_C_1) );

    /* Free Device memory 2 */
    gpuCheck( hipFree(d_A_2) );
    gpuCheck( hipFree(d_B_2) );
    gpuCheck( hipFree(d_C_2) );

    /* Free Device memory 3 */
    gpuCheck( hipFree(d_A_3) );
    gpuCheck( hipFree(d_B_3) );
    gpuCheck( hipFree(d_C_3) );

    gpuCheck( hipStreamDestroy(stream1) );
    gpuCheck( hipStreamDestroy(stream2) );
    gpuCheck( hipStreamDestroy(stream3) );

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
