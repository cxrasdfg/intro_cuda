#include "utils.h"

/**
*** implementation of A x B  ***
*/
extern void fmm_cu(
    const float* const A,const float* const B,
    float* const C,size_t aM,size_t aN, size_t bN);

void fmm(
    const float* const A,const float* const B,
    float* const C,size_t aM,size_t aN, size_t bN){ 
    for (size_t i=0;i<aM;i++){
        for (size_t j=0;j<bN;j++){
            for (size_t p=0;p<aN;p++){
                C[i*bN+j]=A[i*aN+p]*B[p*bN+j];
            }
        }
    }
}

void mat_print(const float * const _data,size_t m,size_t n){
    for(size_t i =0;i<m;i++){    
        for (size_t j=0;j<n;j++){
            printf("%.2f \t", _data[i*n+j]);
        }
        printf("\n");
    }
}

void mat_cmp(const float * const _a,const float * const _b,size_t m,size_t n,float _eps=1e-6){
    for(size_t i =0; i<m;i++){
        for (size_t j=0;j<n;j++){
            size_t idx=i*n+j;
            if (_a[idx]-_b[idx]>_eps){
                printf("eval failed, pair (%f,%f) in (%d,%d)",_a[idx],_b[idx], i,j);
                exit(0);
            }
        }
    }
    printf("eval pass...\n");
}
void mat_init(float * const _data,size_t m,size_t n,int _bval=0){
    for(size_t i =0; i<m;i++){
        for (size_t j=0;j<n;j++){
            size_t idx=i*n+j;
            _data[idx]=idx+_bval;
        }
    }
}

void mat_zero(float * const _data,size_t m,size_t n){
    for(size_t i =0; i<m;i++){
        for (size_t j=0;j<n;j++){
            size_t idx=i*n+j;
            _data[idx]=0;
        }
    }
}
int main(int argc, char ** argv) {
	const int aM=33;
    const int aN=5;
    const int bN=10;
	const int SIZE_A = sizeof(float)*aM*aN;
    const int SIZE_B = sizeof(float)*aN*bN; 
    const int SIZE_C = sizeof(float)*aM*bN;

	// generate the input array on the host
	float h_A[SIZE_A]={0};
    float h_B[SIZE_B]={0};
	float h_C[SIZE_C]={0};
    float h_out[SIZE_C]={0};
    
    mat_init(h_A,aM,aN,0);
    mat_init(h_B,aN,bN,1);
    printf("A:\n");
    mat_print(h_A,aM,aN);
    printf("B:\n");
    mat_print(h_B,aN,bN);
    printf("Results of A x B:\n");
    fmm(h_A,h_B,h_C,aM,aN,bN);
    mat_print(h_C,aM,bN);

    printf("******  use cuda  *******\n");
	// declare GPU memory pointers
	float * d_A;
	float * d_B;
    float * d_C;

	// allocate GPU memory
	cudaMalloc((void**) &d_A, SIZE_A);
	cudaMalloc((void**) &d_B, SIZE_B);
	cudaMalloc((void**) &d_C, SIZE_C);

    //mat_zero(h_C,aM,bN);
	// transfer the array to the GPU
	cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_out, SIZE_C, cudaMemcpyHostToDevice);

	// launch the kernel
	// cube<<<1, ARRAY_SIZE>>>(d_out, d_in);
    fmm_cu(d_A,d_B,d_C,aM,aN,bN);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_C, SIZE_C, cudaMemcpyDeviceToHost);
    mat_print(h_out,aM,bN);
    mat_cmp(h_C,h_out,aM,bN);

	cudaFree(d_A);
	cudaFree(d_B);
    cudaFree(d_C);

	return 0;
}
