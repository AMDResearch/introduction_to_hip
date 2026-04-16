# Scale and Shift Kernel

This program takes an input array and scales (multiplies) and shifts (adds to) each element by `scale` and `shift`, respectively.

Your task is to add the parts of the code that:
* Allocate GPU memory for the arrays
* Copy initial values from the host arrays to the device arrays
* Complete the kernel
* Launch the kernel
* Copy the results back from the device (GPU) to the host (CPU).

Look for the TODOs in the code.

To compile and run:
```
$ make

$ sbatch submit.sh
```
