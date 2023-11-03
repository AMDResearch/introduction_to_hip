# Add the device-to-host data transfer

This example simply initializes an array of integers to 0 on the host, sends the 0s from the host array to the device array, then adds 1 to each element in the kernel, then sends the 1s back to the host array.

However, the device-to-host data transfer call (`hipMemcpy`) is missing. Please add in the missing call and run the program.
