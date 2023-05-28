#
# makdefile for project 1, csc596-192
# a minicourse on blockchain
#
# author: bjr
# date: 17 jan 2018
#

#A= sm_35
A= sm_70

P= vector-add
Q= inner-prod

VERBOSE= -DIS_VERBOSE
N_ELEM= -DN_ELEM=7

all:
	make clean
	make test-add
	make test-prod

$P: $P.cu
	nvcc -arch $A ${N_ELEM} ${VERBOSE} -lm -o $P $P.cu

test-add: $P
	./$P

$Q: $Q.cu
	nvcc -arch $A ${N_ELEM} ${VERBOSE} -lm -o $Q $Q.cu

test-prod: $Q
	./$Q

clean:
	-rm $P $Q
