# this is for building the OpenCL example contained within
# [-x cu] argument swaps the .cl file extensions for .cu
/usr/local/cuda-10.0/bin/nvcc -x cu cl_test.cl -o driver -lOpenCL
