#include <cstdio>
#include <omp.h>

const int TAB_SIZE = 10000000;

static int tab[TAB_SIZE];

static void init_tab(); // initialize tab
static int sums(); // sequential sum
static int sump(); // omp sum

int main(int argc, char* argv[]) {
  init_tab();

  printf("sum = %d\n", sums());

  printf("sum = %d\n", sump());

  return 0;
}

void init_tab() {
  for (int i=0; i<TAB_SIZE; i++) {
    tab[i] = i % 100000;
  }
}

int sums() {
  int sum = 0;
  for (int i=0; i<TAB_SIZE; i++) {
    sum = sum + tab[i];
  }
  return sum;
}

int sump() {
  int sum = 0;

#pragma omp parallel for reduction(+:sum)
  for (int i=0; i < TAB_SIZE; i++) {
    sum = sum + tab[i];
  }

  return sum;
}

