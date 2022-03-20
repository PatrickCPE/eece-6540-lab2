__kernel void calc_pi(int num_iterations, __global float* calc_buff, __global float* res_buf, int num_workers) {


  // Assign each worker their portion of the calculation
  for (int i = 0; i < 10; i++){
    // pi/4 = sum([(-1^(n))(1/(2*n + 1))], 0, num_iterations*work_units) <= Pi formula

    // Each worker will do two sets of the math for 2 digits of precision
    if(get_group_id(0) == 0){
      int curr_index = ((int)get_local_id(0) * 16) + i;
      if (index % 2){
        calc_buff[curr_index] = -4.0 / (2.0 * (float)(curr_index) + 1.0);
        printf("local thread:%d Writing to index:%d\n", curr_index, curr_index+1);
      } else {
        calc_buff[curr_index] = 4.0 / (2.0 * (float)(curr_index) + 1.0);
        printf("local thread:%d Writing to index:%d\n", curr_index, curr_index+1);
      }
    }
  }


if(get_global_id(0) % 5 == 0){
  //printf("global id:%d local_id:%d group:%d num_groups:%d \n", (int)get_global_id(0), (int)get_local_id(0), (int)get_group_id(0), (int)get_num_groups(0));
 }
//And when the results have been gathered use a single worker to produce the final result
barrier(CLK_GLOBAL_MEM_FENCE);

int gid = (int)get_global_id(0);
if(gid == 0){
  for (int x = 0; x < num_iterations; x++){
    *res_buf += calc_buff[x];
  }
  printf("done\n");
 }
barrier(CLK_GLOBAL_MEM_FENCE);
}
