#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <Windows.h>
#include <omp.h>

#define CACHE_LINE 64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#define CACHE_LINES_ON_PAGE 512

// ROWS * COLS = 1<<28 = 256 MB (2^8 * 2^10 * 2^10)
#define ROWS ((size_t) 1<<24)
#define COLS ((size_t) 1<<4)

CACHE_ALIGN int tab[ROWS][COLS];
double start;
HANDLE thread_handle = GetCurrentThread();

inline void start_clock() {
  start = (double) clock() / CLK_TCK;
}

inline void print_elapsed_time(const char* operation_name) {
  double elapsed;
  double resolution;

  elapsed = (double) clock() / CLK_TCK;
  resolution = 1.0 / CLK_TCK;

  printf("%s: %8.4f sec\n", operation_name, elapsed-start);
}

void init_tab() {

  start_clock();
  int i;
#pragma omp parallel for shared(tab) private(i)
  for (i=0; i<ROWS; i++) {
    for (int j=0; j<COLS; j++) {
      tab[i][j] = rand();
    }
  }
  print_elapsed_time("init");
}

typedef int (*function)();
inline void run_function(function f, const char* operation_name) {
  printf("-------------------------------------------\n");
  start_clock();
  int sum = f();
  printf("sum: %d\n", sum);
  print_elapsed_time(operation_name);
}

__declspec(noinline) int sum_ij() {
  int sum = 0;
  for (int i=0; i<ROWS; i++) {
    for (int j=0; j<COLS; j++) {
      sum += tab[i][j];
    }
  }
  return sum;
}

__declspec(noinline) int sum_ji() {
  int sum = 0;
  for (int j=0; j<COLS; j++) {
    for (int i=0; i<ROWS; i++) {
      sum += tab[i][j];
    }
  }
  return sum;
}

__declspec(noinline) int sum_sec() {
  int sum = 0;
  for (int j=0; j<COLS; j++)  {
    for (int k=0; k<CACHE_LINES_ON_PAGE; k++) {
      for (int i=k; i<ROWS; i+=CACHE_LINES_ON_PAGE) {
        sum += tab[i][j];
      }
    }
  }
  return sum;
}

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

__declspec(noinline) int sum_par_ij() {
  int sum = 0;
  int i;
#pragma omp parallel for default(none) shared(tab) private(i) reduction(+:sum)
  for (i=0; i<ROWS; i++) {
    for (int j=0; j<COLS; j++) {
      sum += tab[i][j];
    }
  }
  return sum;
}

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

int main(int argc, char* argv[]) {

  omp_set_num_threads(4);

#pragma omp parallel
  {

    #pragma omp single
    {
      printf("num threads: %d\n", omp_get_num_threads());
    }

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

  init_tab();

  run_function(sum_ij, "sum_ij");
  run_function(sum_ji, "sum_ji");
  run_function(sum_sec, "sum_sec");
  run_function(sum_pf, "sum_pf");
  run_function(sum_par_ij, "sum_par_ij");
  run_function(sum_par_ji, "sum_par_ji");

  system("pause");
}
