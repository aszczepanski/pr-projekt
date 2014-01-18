el_type sum_omp_reduction2() {
  int j;
  el_type sum = 0;

#pragma omp parallel for default(none) shared(tab) private(j) reduction(+:sum)
  for (j = 0; j < 32; j++) {
    for (size_t i=0; i<TAB_SIZE; i+=32) {
      int k = i + j;
      sum += tab[k];
    }
  }

  return sum;
}
