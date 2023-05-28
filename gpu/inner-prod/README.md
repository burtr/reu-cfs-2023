## Inner product
 
 On a GPU calculate the inner product of two vectors.
 
 First calculate a[i] * b[i] in parallel, one thread for each i, and then sum using a log n method.
 
 See the [dot product](https://www.cs.miami.edu/home/burt/learning/csc596.231/proj2/) project on the GPU mini-course.
 
