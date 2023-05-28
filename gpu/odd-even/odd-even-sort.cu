#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>
#include<time.h>

/*
 * even odd sort
 * author: bjr
 * date: 6 feb 2019
 * last-update:
 *	28 may 2022 -bjr
 *
 */


#ifndef  N_ELEM
#define N_ELEM 8
#endif

#ifndef IS_VERBOSE
#define IS_VERBOSE 0
#endif

#define USAGE_MESSAGE "odd-even-sort"

__shared__ int is_verbose ;

__device__ void swap(int * a, int i) {
	int t ;

	if (a[i]>a[i+1]) {
		t = a[i] ;
		a[i] = a[i+1] ;
		a[i+1] = t ;
	}
}

__global__ void transposition_stage(int * a, int is_even, int n) {
	int thread = threadIdx.x ;

	is_verbose = IS_VERBOSE ;
	if (is_verbose) {
		printf("thread %d, a[%d] = %d\n", thread, thread, a[thread]) ; 
	}

	if (thread+1<n) {
		if (is_even == !(thread%2)) swap(a,thread) ;
	}	
	return ;
}

// *****************************************
// HOST
// *****************************************


int test_array(int * a, int k) {
	int i ;
	for (i=1;i<k;i++) {
		if (a[i-1]>a[i]) return 0 ;
	}
	return 1 ;
}

void print_numbers(const char * s, int * a_i, int n) {
	int i ; 
	printf("%s:\t",s ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf(" (%d numbers)\n",n) ;
	return;
}

void make_random_array(int * a, int n) {
	int i ;
        time_t t ;
	srand((unsigned) time(&t));
	for (i=0;i<n;i++) a[i] = rand()%(n*2) ;
}

int main(int argc, char * argv[]) {
	
	int numbers[N_ELEM] ;
	int n_num ;
	int n_bytes ;
	int stage ;

	int dev = 0 ;
	int * d_a ;

	cudaSetDevice(dev) ;

	assert(N_ELEM%2==0) ;

	n_num = N_ELEM ;
	make_random_array(numbers, n_num) ;
	print_numbers("In", numbers, n_num) ;

	n_bytes = n_num * sizeof(int) ;
	cudaMalloc((int **)&d_a, n_bytes) ;
	cudaMemcpy(d_a, numbers, n_bytes, cudaMemcpyHostToDevice) ;
	
	for (stage=0;stage<n_num;stage++) {
		transposition_stage<<<1,n_num>>>(d_a, (stage%2==0), n_num) ;
	}

	cudaMemcpy(numbers, d_a, n_bytes, cudaMemcpyDeviceToHost) ;
	print_numbers("Out", numbers, n_num) ;
	if (test_array(numbers, n_num)==1) 
		printf("the sort is correct!\n") ;
	else printf("the sort is NOT correct!\n") ;
	cudaFree(d_a) ;

	return 0 ;
}

