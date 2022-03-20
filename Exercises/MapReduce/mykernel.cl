__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf, int num_workers) {



  // Assign each worker their portion of the calculation
  for (int i = 0; i < num_workers; i++){
    // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula

    // Each worker will do two sets of the math for 2 digits of precision
    if(get_global_id(0) == i){
        calc_buff[i * 2] = 4.0 / (2.0 * (float)(i * 2) + 1.0);
        calc_buff[i * 2 + 1] = -4.0 / (2.0 * (float)(i * 2 + 1) + 1.0);
    }
  }

  //And when the results have been gathered use a single worker to produce the final result
  barrier(CLK_GLOBAL_MEM_FENCE);

  int gid = (int)get_global_id(0);
  if(gid == num_workers - 1){
    for (int x = 0; x < num_workers * 2; x++){
      *res_buf += calc_buff[x];
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
}
