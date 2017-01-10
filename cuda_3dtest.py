import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule('''
#include <stdio.h>
__global__ void print_3d(char* in)
{
    int i;
    for(i=0;i<60;i++)
        printf("%d ", (int)in[i]);
    printf("\\n");
}
''')

print_3d = mod.get_function("print_3d")
inp = np.zeros((3, 4, 5), np.uint8)   # Initialized to black image. Add color later
inp[0,0,0]=1
inp[0,0,1]=2
inp[0,1,0]=3
inp[1,0,0]=4

print_3d(drv.In(inp), grid=(1,1), block=(1,1,1))
