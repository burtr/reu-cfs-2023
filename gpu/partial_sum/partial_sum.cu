#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<unistd.h>
#include<assert.h>

/*
 * partial sum on a GPU
 * last update:
 *	28 may 2023: burt
 */


#ifndef N_ELEM
#define N_ELEM 64
#endif

#define USAGE_MESSAGE "usage: %s [-v] n\n"

// cuda kernels

__global__ void collect_array(float *c, int k) {
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	int j ;
#ifdef IS_VERBOSE 
	printf("(%s,%d): thread %d\n", __FILE__,__LINE__,i) ;
#endif
	if (i&k) {
		k -= 1 ;
		j = (i&(~k))-1; 
		c[i] += c[j] ;
	}
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

int power_of_two(int x) {
	while (! (x&1) ) x /= 2 ;
	return x==1 ;
}

float array_diff(float *a, float *b, int n ) {
	float s = 0.0 ;
	float t ;
	int i ;
	for (i=0;i<n;i++) {
		t = a[i]-b[i] ;
		if (t<0.0) t = - t ;
		s += t ; 
	}
	return s; 
}


float *  partial_sum_cpu(float * a, float *b, int n ) {
	int i ; 
	a[0] = b[0] ;
	for (i=1;i<n;i++) a[i] = a[i-1] + b[i] ;
	return a ;
} 

int main(int argc, char * argv[]) {
	int dev = 0 ;
	int n_bytes, n_elem ;
	float * h_a, * h_b, * h_c ;
	float * d_a ;
	int ch, is_verbose ;
	char * command = strdup(argv[0]) ;

	is_verbose = 0 ;
	while ((ch = getopt(argc, argv, "v")) != -1) {
		switch(ch) { 
		case 'v':
			is_verbose += 1 ; 
			break ;
		case '?': 
		default: 
			printf(USAGE_MESSAGE, command) ;
			return 0 ; 
		} 
	} 
	argc -= optind; 
	argv += optind;
	if ( argc!= 1 ) {
		fprintf(stderr, USAGE_MESSAGE, command ) ; 
		exit(0) ; 
	}
	n_elem = atoi(argv[0]) ;
        assert(power_of_two(n_elem));
	n_bytes = n_elem * sizeof(float) ;

	cudaSetDevice(dev) ;

	h_a = (float *) malloc(n_bytes) ;
	h_b = (float *) malloc(n_bytes) ;
	h_c = (float *) malloc(n_bytes) ;

	cudaMalloc((float **)&d_a, n_bytes) ;
	initialData(h_a, n_elem ) ;

	partial_sum_cpu(h_c, h_a, n_elem) ;

	cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice) ;

	
	{ 
		int k = 1 ;
		while (k<n_elem) {
			collect_array <<<1,n_elem>>> ( d_a, k ) ;
			k *= 2 ;
		}
	}

	cudaMemcpy(h_b, d_a, n_bytes, cudaMemcpyDeviceToHost) ;

	printData("a =", h_a,n_elem) ;
	printData("b =", h_b,n_elem) ;
	printf("error: %f\n", array_diff(h_b,h_c,n_elem)) ;

	cudaFree(d_a) ;
	free(h_a) ;
	free(h_b) ;
	free(h_c) ;
	return 0 ;
}

