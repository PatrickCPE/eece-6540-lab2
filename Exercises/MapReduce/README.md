

# Setup
```shell
ssh devcloud
devcloud_login # 1 then 1
tools_setup # 5
cd eece-6540-lab2/Exercises/MapReduce
mkdir bin
cp kafka.txt bin
```

# Compiling and Running
```shell
cd eece-6540-lab2/Exercises/MapReduce
aoc -march=emulator mykernel.cl -o bin/mykernel.aocx
make
cd bin
./main
```
