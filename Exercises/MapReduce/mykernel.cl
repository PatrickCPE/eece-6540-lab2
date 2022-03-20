__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf) {

  /* Make sure local processing has completed */
  //barrier(CLK_GLOBAL_MEM_FENCE);

  /* Perform global reduction */
  int gid = get_global_id(0);
  int used_workers = 0;
  int num_runs = 0;
  int global_size = (int)get_global_size(0);
  //Determine how many workers we have compared to desired num of iterations
  if (num_iterations < global_size){
    used_workers = num_iterations;
    num_runs = 1;
  } else {
    used_workers = global_size;
    num_runs = num_iterations / global_size + 1; // Add 1 to ensure round up, handle in flow control
  }

  for (int i = 0; i < used_workers; i++){
    if (get_global_id(0) == i){
      for (int j = 0; j < num_runs; j++){
        // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula
        int curr_iter = (i * num_runs + j);

        if (curr_iter < num_iterations){ // Handles the rounding on the integer for num runs
          if ((curr_iter) % 2){ // Negative iteration
            calc_buff[curr_iter] = -4.0 / (2.0 * (float)curr_iter + 1.0);
          } else { // Positive iteration
            calc_buff[curr_iter] = 4.0 / (2.0 * (float)curr_iter + 1.0);
          }
        }
      }
    }
  }



  //And when the results have been gathered use a single worker to produce the final result
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(gid == 0){
    for (int x = 0; x < num_iterations; x++){
      *res_buf += calc_buff[x];
    }
  }
}
