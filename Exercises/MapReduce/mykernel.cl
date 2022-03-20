__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf, int num_workers) {



  // Assign each worker their portion of the calculation
  for (int i = 0; i < num_workers; i++){
    // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula

    if(get_global_id(0) == i){
      for (int j = 0; j < num_iterations; j++){
        int curr_index = i * num_iterations + j;
        if ((i) % 2){ // Negative iteration
          calc_buff[curr_index] = -4.0 / (2.0 * (float)curr_index + 1.0);
        } else { // Positive iteration
          calc_buff[curr_index] = 4.0 / (2.0 * (float)curr_index + 1.0);
        }
      }
    }
  }

  //And when the results have been gathered use a single worker to produce the final result
  barrier(CLK_GLOBAL_MEM_FENCE);

  int gid = (int)get_global_id(0);
  if(gid == num_workers - 1){
    for (int x = 0; x < num_iterations; x++){
      *res_buf += calc_buff[x];
    }
  }
}
