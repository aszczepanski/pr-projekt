#pragma omp parallel
{
  const int processor_count = 4;
  int th_id=omp_get_thread_num();

  DWORD_PTR mask = (1 << (th_id % processor_count));
  DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);

  if (result==0) {
    printf("error SetThreadAffnityMask\n");
  }

  else {
    printf("previous mask for thread %d : %d\n",th_id,result);
    printf("new mask for thread %d : %d\n",
      th_id,SetThreadAffinityMask(thread_handle, mask));
  }
}
