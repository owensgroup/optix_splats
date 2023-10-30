# optix8_template
This repository can be forked if you are planning on building something with Optix8. 
This will compile the saxpy example from the RTX Compute Samples. If you would like
to write your own RTX pipeline functions you can fork this repo and modify kernels.cu
in the rtx folder.

## Compilation
You can compile this code with the following commands
```
mkdir build && cd build
cmake .. -DOPTIX_HOME={"Path to Optix8"}
make -j
```
