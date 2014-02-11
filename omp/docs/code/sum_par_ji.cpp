__declspec(noinline) int sum_par_ji() {
  int sum = 0;
  int j;
#pragma omp parallel for default(none) shared(tab) private(j) reduction(+:sum)
  for (j=0; j<COLS; j++) {
    for (int i=0; i<ROWS; i++) {
      sum += tab[i][j];
    }
  }
  return sum;
}
