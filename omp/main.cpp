#include <omp.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

#define TAB_SIZE 100000000

typedef unsigned long long el_type;

el_type tab[TAB_SIZE];

// initialize tab
void init_tab() {
	for (size_t i=0; i<TAB_SIZE; i++) {
		tab[i] = static_cast<el_type>(i)/1000;
	}
}

// sequential sum
el_type sums() {
	el_type sum = 0;

#pragma omp parallel 
#pragma omp single
	{
	for (size_t i=0; i<TAB_SIZE; i++) {
		sum += tab[i];
	}
	}

	return sum;
}

// omp sum
el_type sum_omp_reduction() {
	size_t i;
	el_type sum = 0;

#pragma omp parallel for default(none) shared(tab) private(i) reduction(+:sum) // schedule(dynamic,8192)
	for (i=0; i<TAB_SIZE; i++) {
		sum += tab[i];
	}

	return sum;
}

int main(int argc, char* argv[]) {
#pragma omp parallel
	{
#pragma omp single
		{
			cout << "num threads: " <<  omp_get_num_threads() << endl;
		}
	}

	init_tab();
	
	steady_clock::time_point start = steady_clock::now();

	cout << sums() << endl;

	steady_clock::time_point t1 = steady_clock::now();

	cout << sum_omp_reduction() << endl;

	steady_clock::time_point t2 = steady_clock::now();

	cout << duration_cast<microseconds>(t1-start).count() << "us\n";
	cout << duration_cast<microseconds>(t2-t1).count() << "us\n";

	// system("pause");
	return 0;
}
