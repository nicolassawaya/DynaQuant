#!/bin/bash

# PLEASE READ:
# CULA tools (culatools.com) is required. A free academic liscence is available from the site.

nvcc propagate.cu -I. -lcufft -lcusparse -lcublas -lcurand -lcula_lapack -lcudart -I/usr/local/cula/include -L/usr/local/cula/lib64 -arch sm_21


