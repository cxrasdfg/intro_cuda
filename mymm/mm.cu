#include "utils.h"

__global__ 
void __kernel_mm__(
    const float * const A, const float * const B,
    float * const C,size_t aM,
    size_t aN, size_t bN){
    //int bsw=gridDim.x;
    //int bsh=gridDim.y;
    int bsi=blockIdx.y;
    int bsj=blockIdx.x;
    int tsw=blockDim.x;
    int tsh=blockDim.y;
    int tsi=threadIdx.y;
    int tsj=threadIdx.x;

    int beginRow=bsi*tsh+tsi;
    int beginCol=bsj*tsw+tsj;
    
    float sum=.0;
    //printf("the dim y:%d",gridDim.y);
    //return;
    if (beginRow < aM && beginCol<bN){
        //printf("%d ", bsi);
        int idx=beginRow*bN+beginCol;
        //printf("%d ", idx);
        for(int i=0;i<aN;i++){
            
            sum+=(A[beginRow*aN+i])*(B[i*bN+beginCol]); 
            //printf("%.2f ",A[beginRow*aN+i]);
            //printf("%.2f ",C[idx]);
        }
        C[idx]=sum;
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
    dim3 grid_dim(ceil(float(bN)/block_size),ceil(float(aM)/block_size),1); 
    //printf("%d %d",grid_dim.x,grid_dim.y);
    //exit(0);
    __kernel_mm__<<<grid_dim,block_dim>>>(A,B,C,aM,aN,bN);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
