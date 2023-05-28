#
# make for for bitonic sort on a GPU
# last-update:
#	28 may 2023 -bjr
#

#A= sm_30
A= sm_50
N=16

# verbose is 1
#V=0
V=1

all:
	make clean
	make bitonic-sort-gpu
	make test-gpu
	make bitonic-sort-cpu
	make test-cpu

bitonic-sort-gpu: bitonic-sort-gpu.cu
	nvcc -arch $A -DN_ELEM=$N -o bitonic-sort-gpu bitonic-sort-gpu.cu

bitonic-sort-cpu: bitonic-sort-cpu.c
	cc -DN_ELEM=$N -o $@ $<

test-gpu: bitonic-sort-gpu
	./bitonic-sort-gpu

test-cpu: bitonic-sort-cpu
	./bitonic-sort-cpu

clean:
	-rm bitonic-sort-cpu bitonic-sort-gpu
