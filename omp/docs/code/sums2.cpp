el_type sums2() {
  el_type sum = 0;

  for (int j = 0; j < 32; j++) {
    for (size_t i=0; i<TAB_SIZE; i+=32) {
      int k = i + j;
      sum += tab[k];
    }
  }

  return sum;
}
