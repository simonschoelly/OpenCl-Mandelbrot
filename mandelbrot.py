#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from PIL import Image

WIDTH = 1152
HEIGHT = 768
NUM_ITERATIONS = 100

# If the environment variable PYOPENCL_CTX is set, pyopencl will automatically choose a device.
# import os
# os.environ["PYOPENCL_CTX"] = "0"


# this kernel calculates in parallel for each point c of a complex
# grid, if the sequence z_0 = c, z_{n+1} = {z_n}^2 + c diverges 
# for n -> infinity. For each point, we store 0 in buffer out if
# if it does not diverge (i.e belongs to the Mandelbrot set) or the
# number of iterations after which we are sure that the sequence diverges.
KernelSource = '''
__kernel void mandelbrot(const int WIDTH,
                         const int HEIGHT,
                         const int NUM_ITERATIONS,
                         __global int *out
                        )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    double c_real = x * 3.0 / (WIDTH - 1) - 2.0;
    double c_imag = y * 2.0 / (HEIGHT - 1) - 1.0;

    double z_real = 0.0;
    double z_imag = 0.0;
    double tmp_z_real;
    double norm;

    int divergence_at = 0;
    for (int i = 1; i <= NUM_ITERATIONS; ++i) {
        tmp_z_real = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2 * z_real * z_imag                   + c_imag;
        z_real = tmp_z_real;

        // if norm > 4.0, we can be sure that the sequence diverges
        norm = z_real * z_real + z_imag * z_imag;
        if (norm > 4.0) {
            divergence_at = i;
            break;
        }
    }

    out[y * WIDTH + x] = divergence_at;
}
'''

def main():
    # setup open-cl queue and context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    
    # setup an uninitialized buffer on the device
    d_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                        WIDTH * HEIGHT * np.dtype(np.float64).itemsize)

    # compile the Mandelbrot kernel
    program = cl.Program(context, KernelSource).build()
    mandelbrot = program.mandelbrot
    mandelbrot.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None])

    # run the Mandelbrot kernel 
    globalrange = (WIDTH, HEIGHT)
    localrange = None
    mandelbrot(queue, globalrange, localrange, WIDTH, HEIGHT, NUM_ITERATIONS, d_out)
    queue.finish();

    # copy the buffer from the device to the host
    h_out = np.empty((HEIGHT, WIDTH), dtype=np.int32)
    cl.enqueue_copy(queue, h_out, d_out)

    # display the buffer as an image
    img = 255.0 * (h_out / np.max(h_out))
    Image.fromarray(img).show()


if __name__ == "__main__":
    main()
