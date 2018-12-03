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

/***
*** enable the shared memory to calculate ***
*/
template <int block_size>
__global__ 
void __kernel_mm_shared__(
    const float * const A, const float * const B,
    float * const C,size_t aM,
    size_t aN, size_t bN,
    size_t numSM){
    //int bsw=gridDim.x;
    //int bsh=gridDim.y;
    int bsi=blockIdx.y;
    int bsj=blockIdx.x;
    const int tsw=blockDim.x;
    const int tsh=blockDim.y;
    int tsi=threadIdx.y;
    int tsj=threadIdx.x;

        
    int subi=0;
    int sAj=0;
    int sBi=0;
    int i=0;

    __shared__ float sA[block_size][block_size];
    __shared__ float sB[block_size][block_size];

    int beginRow=bsi*tsh+tsi;
    int beginCol=bsj*tsw+tsj;
    
    float sum=.0;
       
    //printf("the dim y:%d",gridDim.y);
    //return;
    if (beginRow < aM && beginCol<bN){
        //printf("%d ", bsi);
        int idx=beginRow*bN+beginCol;
        for(subi=0;subi<numSM;subi++){
            // copy data from `global` mem to `shared`
            //sAi=beginRow; // i index of sub matrix A in global mem;
            sAj=subi*tsw+tsj; // j index of sub matrix A in global mem;
            sBi=subi*tsh+tsi; // i index of sub matrix B in global mem; 
            //sAj=beginCol; // j index of sub matrix B in global mem;
            if (sAj < aN)
                sA[tsi][tsj]=A[beginRow*aN+sAj];
            else
                sA[tsi][tsj]=.0;

            if (sBi < aN)
                sB[tsi][tsj]=B[sBi*bN+beginCol];
            else
                sB[tsi][tsj]=.0;

            __syncthreads();
            
            for(i=0;i<block_size;i++){
                sum+=sA[tsi][i]*sB[i][tsj];
            }

            __syncthreads();
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
    const int block_size=32;
    dim3 block_dim(block_size,block_size,1);
    dim3 grid_dim(ceil(float(bN)/block_size),ceil(float(aM)/block_size),1); 
    size_t numSM=ceil(float(aN)/block_size);
    //printf("%d %d",grid_dim.x,grid_dim.y);
    //exit(0);
    //__kernel_mm__<<<grid_dim,block_dim>>>(A,B,C,aM,aN,bN);
    __kernel_mm_shared__<block_size><<<grid_dim,block_dim>>>(A,B,C,aM,aN,bN,numSM);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
