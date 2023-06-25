#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<assert.h>
#include<time.h>

#define DOWN 1
#define UP -1
#define FALSE 0
#define TRUE 1


#ifndef N_ELEM
#define N_ELEM 32
#endif

/*
 * bitonic sorting network, iterative
 * author: bjr
 * date: 30 oct 2018
 *
 * This code implements the biotonic sorting.
 * Input will be round up to a power of 2.
 *
 * The diagram shows 3 types of blocks in the sorting network:
 *   blue, green and red.
 * In phase1, various levels begin with two numbers at time, then
 * 4, then 8. The blocks alternative blue, green, blue, green. 
 * The last level has just a blue block.
 *
 * A red block is the fine structure inside either a blue or green
 * block. They are recursive and all go the same direction (no alternation).
 * If this direction is DOWN, they are implementing a blue block, if
 * UP they are implementing a gree block.
 *
 */

static int is_2power(int n) {
	while (!(n&1)) {
		n/=2 ;
	}
	return n==1 ;
}

// bitonic sorting

void swap(int *i1, int *i2, int d ){
	if (d==UP) {
		if (*i1<*i2) {	
			int t ;
			t = *i1 ;
			*i1 = *i2 ;
			*i2 = t ;
		}
	} else { // DOWN
		if (*i1>*i2) {
			int t ;
			t = *i1 ;
			*i1 = *i2 ;
			*i2 = t ;
		}
	}
	return ;
}

void sort_red_block(int * a, int n, int direction) {
	int i ;

	if (n<2) return ;
	for (i=0;i<n/2;i++) {
		swap(a+i,a+i+n/2,direction) ;
	}
	sort_red_block(a, n/2, direction) ;
	sort_red_block(a+n/2, n/2, direction ) ;
	return ;
}

void sort_blue_block(int * a, int n) {
	sort_red_block(a,n,DOWN) ;
	return ;
}

void sort_green_block(int * a,int n) {
	sort_red_block(a,n,UP) ;
	return ;
}

void sort_bitonic(int *a, int n) {
	int level = 2 ;
	int loc ;
	assert(is_2power(n)) ;
	while (level<n) {
		loc = 0 ;
		while (loc<n) {
			sort_blue_block(a+loc,level) ;
			sort_green_block(a+loc+level,level) ;
			loc += 2*level ;
		}
		level *= 2 ;
	}	
	assert(level==n) ;
	sort_blue_block(a,n) ;
	return ;
}

// test and main code


#define BUFFER_N 1024
#define SEP_CHAR " \t\n,"
#define MAX_NUMBERS 1024

#define USAGE_MESSAGE "bitonic-sort [-vF] [_01-test-size_]"

int is_verbose = 0 ;

void print_numbers(char * s, int * a_i, int n) {
	int i ; 
	printf("%s:\t",s ) ;
	for (i=0;i<n;i++) printf("%d ", a_i[i]) ;
	printf(" (%d numbers)\n",n) ;
	return;
}

int ipow(int k) { 
	return 1<<k ;
}

int pad_to_2power(int * a, int n) {
	int p2 = 1 ;
	int j ;
	while (p2<n) {
		p2 *= 2 ;
	}
	if (p2==n) return n ;
	assert(p2<=MAX_NUMBERS) ;
	for (j=n;j<p2;j++) {
		a[j] = 0 ;
	}
	return p2 ;
}

int test_array(int * a, int k) {
	int i ;
	for (i=1;i<k;i++) {
		if (a[i-1]>a[i]) return FALSE ;
	}
	return TRUE ;
}

int power_of_two(int x) {
	while (! (x&1) ) x /= 2 ;
	return x==1 ;
}

void make_random_array(int * a, int n) {
	int i ;
	time_t t ;
	srand((unsigned) time(&t));
	for (i=0;i<n;i++) a[i] = rand()%(2*n) ;
}

int main(int argc, char * argv[]){
	
	int ch ;
	int numbers[N_ELEM] ;
	int n_elem = N_ELEM ;

	while ((ch = getopt(argc, argv, "v")) != -1) {
		switch(ch) {
		case 'v':
			is_verbose = 1 ;
			break ;
		default:
			printf("usage: %s\n", USAGE_MESSAGE) ;
			return 0 ;
		}
	}
	argc -= optind;
	argv += optind;

	make_random_array(numbers, n_elem) ;
	print_numbers("In", numbers, n_elem) ;
	sort_bitonic(numbers, n_elem) ;
	print_numbers("Out", numbers, n_elem) ;
	printf("\n") ;
	return 0 ;
}

