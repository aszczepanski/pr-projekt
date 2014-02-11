int tmp;

__declspec(noinline) int sum_pf() {
  int sum = 0;
  int i;
  for (i=0; i<ROWS-1; i++) {
    for (int j=0; j<COLS; j++) {
      sum += tab[i][j];
      tmp = tab[i+1][j];
    }
  }

  for (int j=0; j<COLS; j++) {
    sum += tab[i][j];
  }
  return sum;
}
