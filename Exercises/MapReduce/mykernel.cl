__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf) {

  /* Make sure local processing has completed */
  //barrier(CLK_GLOBAL_MEM_FENCE);

  /* Perform global reduction */
  int gid = get_global_id(0);
  for (int i = 0; i < num_iterations; i++){
    int curr_iter = gid + i;

    // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula
    if (curr_iter % 2){ // Even iteration
      //res_buf[curr_iter] = 4.0 / (2.0 * (float)curr_iter + 1.0);
    } else { // Odd iteration
      //res_buf[curr_iter] = -4.0 / (2.0 * (float)curr_iter + 1.0);
    }
  }

  //Ensure all work units have performed there local calculation of their index
  barrier(CLK_GLOBAL_MEM_FENCE);

  //Use 4 workers to combine results
  size_t glob_size = get_global_size(0);
  if(gid == 0){
  } else if (gid == 1) {
  } else if (gid == 2) {
  } else if (gid == 3) {
  }

  //And when the results have been gathered use a single worker to produce the final result
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(gid == 0){
    *res_buf = (float)get_global_size(0);
  }
}
