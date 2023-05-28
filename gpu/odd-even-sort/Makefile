#
# odd-even sort on a GPU
# last-update:
# 	28 may 2023 -bjr
#
#

#A= sm_30
A= sm_50
N= 32
P= odd-even-sort

all:
	make test

$P: $P.cu
	nvcc -arch $A -DN_ELEM=$N -o $@ $<

test: $P
	./$P $N

clean:
	-rm $P
