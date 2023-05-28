#
# makdefile for project 1, csc596-192
# a minicourse on blockchain
#
# author: bjr
# date: 17 jan 2018
#

#A= sm_35
A= sm_70

P= partial_sum

#VERBOSE= -DIS_VERBOSE
VERBOSE= 
N_ELEM= -DN_ELEM=8

all:
	make clean
	make partial_sum

$P: $P.cu
	nvcc -arch $A ${N_ELEM} ${VERBOSE} -lm -o $P $P.cu

test: $P
	./$P 4
	./$P 8
	./$P 16
	./$P 32

clean:
	-rm $P
