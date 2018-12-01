#include "utils.h"

__global__ 
void __kernel_mm__(
    const float * const A, const float * const B,
    float * const C,size_t aM,
    size_t aN, size_t bN){
    int bsw=gridDim.x;
    int bsh=gridDim.y;
    int bsi=blockIdx.y;
    int bsj=blockIdx.x;
    int tsw=blockDim.x;
    int tsh=blockDim.y;
    int tsi=threadIdx.y;
    int tsj=threadIdx.x;

    int beginRow=bsi*tsh+tsi;
    int beginCol=bsj*tsw+tsj;
    if (beginRow < aM && beginCol<bN){
        int idx=beginRow*bN+beginCol;
        //printf("%d ", idx);
        for(int i=0;i<aN;i++){
            
            C[idx]= (A[beginRow*aN+i])*(B[i*bN+beginCol]); 
            //printf("%.2f ",A[beginRow*aN+i]);
            //printf("%.2f ",C[idx]);
        }
    }
}

void fmm_cu(
    const float* const A, 
    const float* const B,
    float * const C,
    size_t aM,size_t aN,
    size_t bN){
    int block_size=32;
    dim3 block_dim(block_size,block_size,1);
    dim3 grid_dim(ceil(float(aM)/block_size),ceil(float(bN)/block_size),1); 
    __kernel_mm__<<<grid_dim,block_dim>>>(A,B,C,aM,aN,bN);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
