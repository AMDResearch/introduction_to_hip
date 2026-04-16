#include "hip/hip_runtime.h"

#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaError_t hipError_t
#define cudaGetErrorString hipGetErrorString
#define cudaSuccess hipSuccess
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
