# Lab 2
## Heterogenous Computing - EECE.6540
## Patrick Hoey


# Setup on Intel Devcloud
```shell
ssh devcloud
devcloud_login # 1 then 1
tools_setup # 5
# Assumes you have it cloned
cd eece-6540-lab2/Exercises/MapReduce
mkdir bin # Only need to run once
```

# Compiling and Running
```shell
cd eece-6540-lab2/Exercises/MapReduce # It's not really map reduce, it's calculate pi :)
aoc -march=emulator mykernel.cl -o bin/mykernel.aocx
make
cd bin
./main
```

# References
https://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html 
https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html

