__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf, int num_workers) {

  /* Make sure local processing has completed */
  //barrier(CLK_GLOBAL_MEM_FENCE);

  // Assign each worker their portion of the calculation
  for (int i = 0; i < num_workers; i++){
        // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula

          if ((i) % 2){ // Negative iteration
            calc_buff[i] = -4.0 / (2.0 * (float)curr_iter + 1.0);
          } else { // Positive iteration
            calc_buff[i] = 4.0 / (2.0 * (float)curr_iter + 1.0);
          }
  }



  //And when the results have been gathered use a single worker to produce the final result
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(gid == num_workers - 1){
    for (int x = 0; x < num_iterations; x++){
      *res_buf += calc_buff[x];
    }
  }
}
