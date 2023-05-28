#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

/*
 * add vectors in a GPU
 * author: bjr
 * date: nov 2017
 * last update: jan 2019
 */


#ifndef N_ELEM
#define N_ELEM 32
#endif

// cuda kernel

__global__ void sum_array(float * a, float *b, float * c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	a[i] = b[i] + c[i] ;
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

float distance(float * a, float * b, float *c, int n) {
	float f, dist = 0.0 ;
	int i ;
	for (i=0;i<n;i++) {
		f = b[i] + c[i] - a[i] ;
		dist += f*f ;
	}
	return sqrt(dist) ;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n = N_ELEM ;
	int n_bytes = n * sizeof(float) ;
	float * h_a, * h_b, * h_c ;
	float * d_a, * d_b, * d_c ;

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

        sum_array <<<1,n>>> ( d_a, d_b, d_c ) ;

	cudaMemcpy(h_a, d_a, n_bytes, cudaMemcpyDeviceToHost) ;

	printf("n =\t%d\n", n) ;
	printData("b =", h_b,n) ;
	printData("c =", h_c,n) ;
	printData("sum =",h_a,n) ;
	printf("error = %f\n", distance(h_a,h_b,h_c,n) ) ;

	cudaFree(d_a) ;
	cudaFree(d_b) ;
	cudaFree(d_c) ;
	free(h_a) ;
	free(h_b) ;
	free(h_c) ;

	return 0 ;
}

