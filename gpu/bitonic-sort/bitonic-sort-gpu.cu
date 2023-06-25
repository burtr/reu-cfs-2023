#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>
#include<time.h>


/*
 * bitonic sorting network, iterative
 * author: bjr
 * date: 30 oct 2018
 *
 * This code implements the biotonic sorting.
 * Input will be round up to a power of 2.
 *
 * The diagram shows 3 types of boxes in the sorting network: blue, green and red.
 *
 * Blue and green boxes differ in direction. Both contain a recursive cascade of
 * red boxes.
 *
 * Threads are assigned to the array index of same index, and in each round a calculation
 * locates the thread within its box, and determines the box color.
 *
 */


#define GREEN 1
#define BLUE 0

#ifndef N_ELEM
#define N_ELEM 4
#endif

#ifndef IS_VERBOSE
#define IS_VERBOSE 0
#endif

#define TRUE 1
#define FALSE 0

int is_verbose_g = IS_VERBOSE ;

__shared__ int is_verbose ;

__device__ void swap(int * a, int i1, int i2, int d ){
	int t ;
	if (d==GREEN) {
		if (a[i1]<a[i2]) {
			t = a[i1] ;
			a[i1] = a[i2] ;
			a[i2] = t ;
		}
	} else { // BLUE
		if (a[i1]>a[i2]) {
			t = a[i1] ;
			a[i1] = a[i2] ;
			a[i2] = t ;
		}
	}
	return ;
}

__device__ void red_box(int * a, int thread , int box_base, int box_size, int dir) {
	int w = box_size ;
	int i = box_base ;

	if (is_verbose) {
		printf("thread %d: red_box base: %d, size: %d, color: %d\n", thread, box_base, box_size, dir ) ;
	}
	while (w>1) {
		w /= 2 ;
		if ((thread-i)<w) {
			swap(a,thread,thread+w,dir) ;
		}
		else {
			i += w ;
		}
		if (is_verbose) printf("a[%d] = %d\n", thread, a[thread]) ;
		__syncthreads() ;
	}
	return ;
}

__global__ void sort_bitonic(int * a, int n, int is_verbose_h) {
	int thread = threadIdx.x ;
	int box_size ;
	int box_base ;
	int box_color ; 

	is_verbose = is_verbose_h ;

	if (is_verbose) {
		printf("thread %d, a[%d] = %d\n", thread, thread, a[thread]) ; 
	}

	box_size = 2 ;
	while ( box_size<=n ) {
		box_base = thread - (thread%box_size) ;
		box_color = (box_base%(2*box_size)) ? GREEN : BLUE ;
		red_box(a,thread,box_base,box_size,box_color) ;
		box_size *= 2 ;
	}
	return ;
}

int ipow(int k) { 
	return 1<<k ;
}

int test_array(int * a, int k) {
	int i ;
	for (i=1;i<k;i++) {
		if (a[i-1]>a[i]) return FALSE ;
	}
	return TRUE ;
}

void make_random_array(int * a, int n) {
	int i ;
	time_t t ;
	srand((unsigned) time(&t));
	for (i=0;i<n;i++) a[i] = rand()%(2*n) ;
}

int power_of_two(int x) {
	while (! (x&1) ) x /= 2 ;
	return x==1 ;
}

void print_numbers(const char * s, int * a_i, int n) {
	int i ; 
	printf("%s:\t",s ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf(" (%d numbers)\n",n) ;
	return;
}

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int * h_a ;
	int * d_a ;
	int n_elem = N_ELEM ; 
	int n ;
	
	assert(power_of_two(n_elem)) ; 

	cudaSetDevice(dev) ;

	n = n_elem * sizeof(int) ;
	h_a = (int *) malloc(n) ;
	cudaMalloc((int **)&d_a, n) ;

	make_random_array(h_a, n_elem) ;
	print_numbers("test", h_a, n_elem) ;
	cudaMemcpy(d_a, h_a, n, cudaMemcpyHostToDevice) ;
	sort_bitonic <<<1, n_elem>>> ( d_a, n_elem, is_verbose_g ) ;
	cudaMemcpy(h_a, d_a, n, cudaMemcpyDeviceToHost) ;
	print_numbers("sort", h_a, n_elem) ;
	printf("\n") ;
	assert(test_array(h_a,n_elem)==TRUE) ;
	
	cudaFree(d_a) ;
	free(h_a) ;

	return 0 ;
}

