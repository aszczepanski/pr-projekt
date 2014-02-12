for (int j=0; j<COLS; j++)  {
  for (int k=0; k<CACHE_LINES_ON_PAGE; k++) {
    for (int i=k; i<ROWS; i+=CACHE_LINES_ON_PAGE) {
      sum += tab[i][j];
    }
  }
}
