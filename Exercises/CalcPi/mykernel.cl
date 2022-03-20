__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf, int num_workers) {

  //Run enough iterations to get the accuracy we need
  for (int i = 0; i < 100; i++){
    // Equation for Pi
    // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula

    //Computation locked to a single compute group/aka compute device due to the fact floating point
    //math may behave differently on different compute nodes. That would result in too much UB on real systems
    if(get_group_id(0) == 0){
      int curr_index = ((int)get_local_id(0) + i * 16);
      if (curr_index % 2){ // Odd indexes
        calc_buff[curr_index] = -4.0 / ((2.0 * (float)(curr_index)) + 1.0);
      } else { // Even Indexes
        calc_buff[curr_index] = 4.0 / ((2.0 * (float)(curr_index)) + 1.0);
      }
    }
  }

  //Wait for all local workers to complete calculations
  barrier(CLK_LOCAL_MEM_FENCE);

  //Allow local thread 1 to reduce the result into the output buffer
  int gid = (int)get_global_id(0);
  if(gid == 0){
    for (int x = 0; x < 1600; x++){
      res_buf[0] += calc_buff[x];
    }
    printf("done calculation of pi\n");
  }

  //Wait for everything to finish before releasing kernel execution
  barrier(CLK_GLOBAL_MEM_FENCE);
}
