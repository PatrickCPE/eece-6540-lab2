#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef AOCL
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;
void cleanup();
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

int main()
{
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_int ret;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    cl_uint num_comp_units;
    size_t global_size;
    size_t local_size;


    FILE *fp;
    char fileName[] = "./mykernel.cl";
    char *source_str;
    size_t source_size;

#ifdef AOCL  /* Altera FPGA */
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    // Get the OpenCL platform.
    platforms[0] = findPlatform("Intel(R) FPGA Emulation");
    if(platforms[0] == NULL) {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }
    // Query the available OpenCL device.
    getDevices(platforms[0], CL_DEVICE_TYPE_ALL, &ret_num_devices);
    printf("Platform: %s\n", getPlatformName(platforms[0]).c_str());
    printf("Using one out of %d device(s)\n", ret_num_devices);
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("device name=  %s\n", getDeviceName(device_id).c_str());
#else
#error "unknown OpenCL SDK environment"
#endif


    /* Determine global size and local size */
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(num_comp_units), &num_comp_units, NULL);
    printf("num_comp_units=%u\n", num_comp_units);
#ifdef AOCL  /* local size reported Altera FPGA is incorrect */
    local_size = 16;
#endif
    printf("local_size=%lu\n", local_size);
    global_size = num_comp_units * local_size;
    printf("global_size=%lu, local_size=%lu\n", global_size, local_size);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

#ifdef AOCL  /* on FPGA we need to create kernel from binary */
   /* Create Kernel Program from the binary */
   std::string binary_file = getBoardBinaryFile("mykernel", device_id);
   printf("Using AOCX: %s\n", binary_file.c_str());
   program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);
#else
#error "unknown OpenCL SDK environment"
#endif


    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
      printf("Failed to build program.\n");
      exit(1);
    }

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "calc_pi", &ret);
    if (ret != CL_SUCCESS) {
      printf("Failed to create kernel.\n");
      exit(1);
    }

    //Create buffer to hold the intermediate results
    int num_workers = global_size;
    int num_iterations[1] = {2}; // In the summation this is effectively n assuming n starts at 0
    //float *calc = (float *) calloc((num_iterations * num_workers), sizeof(float));
    float calc[1600] = {0.0};
    cl_mem calc_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                        CL_MEM_COPY_HOST_PTR, sizeof(calc), &calc, &ret);
    if(ret < 0) {
       perror("Couldn't create a buffer");
       exit(1);
    };

    //Create a buffer to hold the final result only
    float result[1] = {0};
    cl_mem res_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY |
                                        CL_MEM_COPY_HOST_PTR, sizeof(result), &result, &ret);
    if(ret < 0) {
      perror("Couldn't create a buffer");
      exit(1);
    };



    /* Create kernel argument */
    ret = 0;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&num_iterations); // int num_iterations
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&calc_buffer); // float* calc_buff
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&res_buffer); // float* res_buff
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&num_workers); // int num_workers
    if(ret < 0) {
       printf("Couldn't set a kernel argument");
       exit(1);
    };

    /* Enqueue kernel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size,
          &local_size, 0, NULL, NULL);
    if(ret < 0) {
       perror("Couldn't enqueue the kernel");
       printf("Error code: %d\n", ret);
       exit(1);
    }

    // Wait for processing to complete
    ret = clFinish(command_queue);
    if (ret < 0){
      printf("Error waiting for clFinish\n");
    }

    /* Read and print the result */
    ret = clEnqueueReadBuffer(command_queue, res_buffer, CL_TRUE, 0,
       sizeof(result), &result, 0, NULL, NULL);
    if(ret < 0) {
       perror("Couldn't read the buffer");
       exit(1);
    }

    printf("Final value of pi: %f\n", result[0]);


    /* free resources */

    //clReleaseMemObject(calc_buffer);
    //clReleaseMemObject(res_buffer);
    //clReleaseCommandQueue(command_queue);
    //clReleaseKernel(kernel);
    //clReleaseProgram(program);
    //clReleaseContext(context);

    return 0;
}

#ifdef AOCL
// Altera OpenCL needs this callback function implemented in main.c
// Free the resources allocated during initialization
void cleanup() {
}
#endif
