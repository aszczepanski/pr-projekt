el_type sums() {
  el_type sum = 0;

  for (size_t i = 0; i < TAB_SIZE; i++) {
    size_t& k = i;
    sum += tab[k];
  }

  return sum;
}
