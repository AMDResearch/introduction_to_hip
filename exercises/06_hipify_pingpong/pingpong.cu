#include <stdio.h>

/* Macro for checking GPU API return values */
#define gpuCheck(call)                                                                           \
do{                                                                                              \
    cudaError_t gpuErr = call;                                                                   \
    if(cudaSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr)); \
        exit(1);                                                                                 \
    }                                                                                            \
}while(0)

void host_device_transfer(const char* direction){

    int loop_count = 50;

    for(int i=10; i<=27; i++){

        long int N = 1 << i;

        size_t bytes = N * sizeof(double);

        float milliseconds = 0.0;

        double *h_A;
        gpuCheck( cudaMallocHost(&h_A, bytes) );

        double *d_A;
        gpuCheck( cudaMalloc(&d_A, bytes) );

        cudaEvent_t start, stop;
        gpuCheck( cudaEventCreate(&start) );
        gpuCheck( cudaEventCreate(&stop) );

        for(int j=0; j<N; j++){
            h_A[j] = (double)rand()/(double)RAND_MAX;
        }

        /* Warm-up loop */
        if( strcmp(direction, "H2D") == 0){

            for(int iteration=0; iteration<5; iteration++){
                gpuCheck( cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) );
            }

        }
        else if( strcmp(direction, "D2H") == 0){

            for(int iteration=0; iteration<5; iteration++){
                gpuCheck( cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) );
            }        
        }
        else{
            printf("Error - unknown direction\n");
            exit(1);
        }

        gpuCheck( cudaDeviceSynchronize() );
        gpuCheck( cudaEventRecord(start, NULL) );

        /* Timed loop */
        if( strcmp(direction, "H2D") == 0){

            for(int iteration=0; iteration<loop_count; iteration++){
                gpuCheck( cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) );
            }
        }
        else if( strcmp(direction, "D2H") == 0){

            for(int iteration=0; iteration<loop_count; iteration++){
                gpuCheck( cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) );
            }
        }
        else{
            printf("Error - unknown direction\n");
            exit(1);
        }

        gpuCheck( cudaEventRecord(stop, NULL) );
        gpuCheck( cudaEventSynchronize(stop) );
        gpuCheck( cudaEventElapsedTime(&milliseconds, start, stop) );

        double bandwidth = ( 1000.0 * (double)loop_count * (double)bytes ) / ( (double)milliseconds * 1000.0 * 1000.0 * 1000.0);
        double bytes_mb  = (double)bytes / (1024.0 * 1024.0);

        printf("Buffer Size (MiB): %14.9f, Time (ms): %14.9f, Bandwidth (GB/s): %14.9f\n", bytes_mb, milliseconds, bandwidth);
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[])
{
    printf("----- H2D -----\n");
    host_device_transfer("H2D");

    printf("----- D2H -----\n");
    host_device_transfer("D2H");

    printf("\n__SUCCESS__\n");

    return 0;
}
