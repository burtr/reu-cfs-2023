#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

/*
 * inner product vectors in a GPU
 * author: bjr
 * date: jan 2019
 * last update:
 */


#ifndef N_ELEM
#define N_ELEM 32
#endif

// cuda kernels

__global__ void prod_array(float * a, float *b, float * c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
#ifdef IS_VERBOSE
	printf("(%s,%d): thread %d\n", __FILE__,__LINE__,i) ;
#endif
	a[i] = b[i] * c[i] ;
	return ;
}

__global__ void fold_array(float *c, int k) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
#ifdef IS_VERBOSE 
	printf("(%s,%d): thread %d\n", __FILE__,__LINE__,i) ;
#endif
	c[i] += c[i+k] ;
	return ;
}



// host routines

void initialData(float *ip, int size) {
	time_t t ;
	int i ;
	static int j = 0 ;

	if (!j++) srand ((unsigned)time(&t)) ;
	for (i=0; i<size; i++) {
		ip[i] = (float) ( rand() & 0xFF ) / 10.0f ;
	}
	return ;
}

#define PRINT_I 6
#define PRINT_L 2

void printData(const char * s, float *ip, int n) {
	int i, k ;
	int f = PRINT_I ;
	int l = PRINT_L ;
	printf("%s\t",s) ;
	if (n<=f) {
		for (i=0;i<n;i++) {
			printf("%5.2f ", ip[i]) ;
		}
		printf("\n") ;
		return ;
	}
	for (i=0;i<f;i++) {
		printf("%5.2f ", ip[i]) ;
	}
	printf("\t...\t") ;
	k = n - l ;
	if (k<f) k = f ;
	for (i=k;i<n;i++) {
		printf("%5.2f ", ip[i]) ;
	}
	printf("\n") ;
	return ;
}

float check(float calculated, float * b, float *c, int n) {
	float dot = 0.0 ;
	int i ;
	for (i=0;i<n;i++) {
		dot += b[i]*c[i] ;
	}
	return calculated-dot ;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int n_bytes = n * sizeof(float) ;
	float * h_a, * h_b, * h_c ;
	float * d_a, * d_b, * d_c ;
	int m, k ; 
	float h_result ;

	cudaSetDevice(dev) ;

	h_a = (float *) malloc(n_bytes) ;
	h_b = (float *) malloc(n_bytes) ;
	h_c = (float *) malloc(n_bytes) ;

	cudaMalloc((float **)&d_a, n_bytes) ;
	cudaMalloc((float **)&d_b, n_bytes) ;
	cudaMalloc((float **)&d_c, n_bytes) ;

	initialData(h_b, n ) ;
	initialData(h_c, n ) ;

	// send data to cuda device
	cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice) ;
	cudaMemcpy(d_c, h_c, n_bytes, cudaMemcpyHostToDevice) ;

#ifdef IS_VERBOSE
	printf("(%s,%d): enqueue prod_array with %d threads.\n", __FILE__,__LINE__,n) ;
#endif

        prod_array <<<1,n>>> ( d_a, d_b, d_c ) ;
	m = n ;
	while (m>1)  {
		if (m&0x01) {
			// odd
			k = m / 2 ; // rounds down
#ifdef IS_VERBOSE
	printf("(%s,%d): enqueue fold_array with %d threads.\n", __FILE__,__LINE__,k) ;
#endif
			fold_array <<<1,k>>>(d_a,k+1) ;
			m = k + 1 ;
		}
		else {
			// even
			k = m / 2 ; // exact divide
#ifdef IS_VERBOSE
	printf("(%s,%d): enqueue fold_array with %d threads.\n", __FILE__,__LINE__,k) ;
#endif
			fold_array <<<1,k>>>(d_a,k) ;
			m = k ;
		}
	}

	cudaMemcpy(&h_result, d_a, sizeof(float), cudaMemcpyDeviceToHost) ;

	printf("n =\t%d\n", n) ;
	printData("b =", h_b,n) ;
	printData("c =", h_c,n) ;
	printf("result =\t%f\n", h_result) ;
	printf("error = %f\n", check(h_result,h_b,h_c,n) ) ;

	cudaFree(d_a) ;
	cudaFree(d_b) ;
	cudaFree(d_c) ;
	free(h_a) ;
	free(h_b) ;
	free(h_c) ;

	return 0 ;
}

