el_type sum_omp_reduction() {
  int i;
  el_type sum = 0;

#pragma omp parallel for default(none) shared(tab) private(i) reduction(+:sum)
  for (i=0; i<TAB_SIZE; i++) {
    sum += tab[i];
  }

  return sum;
}
