#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <Windows.h>

#define CACHE_LINE 64
#define CACHE_ALIGN __declspec(align(CACHE_LINE))

HANDLE thread_handle = GetCurrentThread();
double start;

const long long num_steps = 200000000; //1000000000;
double step;

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

inline void set_thread_affinity(const int number_of_threads, const int number_of_cores) {
	omp_set_num_threads(number_of_threads);

#pragma omp parallel
    {
            #pragma omp single
            {
                    printf("num threads: %d\n", omp_get_num_threads());
            }
        
            int th_id = omp_get_thread_num();
            DWORD_PTR mask = (1 << (th_id % number_of_cores));
            DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
            if (result == 0) {
                    printf("error SetThreadAffnityMask\n");
            } else {
                    printf("previous mask for thread %d : %d\n",th_id,result);
                    printf("new mask for thread %d : %d\n",
                            th_id,SetThreadAffinityMask(thread_handle, mask));
            }
    }
}

inline void clear_thread_affinity(const int number_of_threads, const int number_of_cores) {
		omp_set_num_threads(number_of_threads);

#pragma omp parallel
    {
            #pragma omp single
            {
                    printf("num threads: %d\n", omp_get_num_threads());
            }
        
            int th_id = omp_get_thread_num();
            DWORD_PTR mask = ~(1 << number_of_cores);
            DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
            if (result == 0) {
                    printf("error SetThreadAffnityMask\n");
            } else {
                    printf("previous mask for thread %d : %d\n",th_id,result);
                    printf("new mask for thread %d : %d\n",
                            th_id,SetThreadAffinityMask(thread_handle, mask));
            }
    }
}

template <typename Function>
inline void run_function(Function f, const char* operation_name, int* arg = NULL) {
        printf("-------------------------------------------\n");
        start_clock();
		double pi = f(arg);
		if (arg) {
			printf("%d - ", *(int*)arg);
		}
        printf("PI: %15.12f\n", pi);
        print_elapsed_time(operation_name);
}

__declspec(noinline) double pi_serial(int*) {
	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum = sum + 4.0/(1.+ x*x);
	}
	
	pi = sum*step;

	return pi;
}

__declspec(noinline) double pi_par_atomic(int*) {
	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
#pragma omp parallel for private(x)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
#pragma omp atomic
		sum += 4.0/(1.+ x*x);  // TODO atomi
	}
	
	pi = sum*step;

	return pi;
}

__declspec(noinline) double pi_par_reduction(int*) {
	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
#pragma omp parallel for reduction(+:sum) private(x)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum = sum + 4.0/(1.+ x*x);
	}
	
	pi = sum*step;

	return pi;
}

__declspec(noinline) double pi_par_false_sharing(int* start_position_in_array) {
	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
	CACHE_ALIGN volatile double sums[32] = { };
#pragma omp parallel shared(sums,num_steps,start_position_in_array) private(x,i)
	{
		int th_id = omp_get_thread_num();
		if (start_position_in_array) {
			th_id += *start_position_in_array;
		}
#pragma omp for
		for (i=0; i<num_steps; i++) {
			x = (i + .5)*step;
			sums[th_id] = sums[th_id] + 4.0/(1.+ x*x);
		}
	}
	
	for (int i=0; i<32; i++) {
		sum += sums[i];
	}
	
	pi = sum*step;

	return pi;
}

int main(int argc, char* argv[]) {
	printf("sizeof(double) = %d\n", sizeof(double));
	
	set_thread_affinity(4, 4);

	run_function(pi_serial, "serial");
	run_function(pi_par_atomic, "parallel atomic");
	run_function(pi_par_reduction, "parallel reduction");
	run_function(pi_par_false_sharing, "parallel false sharing");

	set_thread_affinity(4, 1);
	run_function(pi_par_false_sharing, "parallel false sharing");

	set_thread_affinity(2, 4);

	for (int start_position = 0; start_position < 20; start_position++) {
		run_function(pi_par_false_sharing, "parallel false sharing", &start_position);
	}

	set_thread_affinity(4, 4);
	for (int i=0; i<3; i++) {		
		run_function(pi_par_reduction, "parallel reduction 4x4");		
		
	}
	set_thread_affinity(4, 2);
	for (int i=0; i<3; i++) {
		run_function(pi_par_reduction, "parallel reduction 4x2");
	}
	clear_thread_affinity(4, 4);
	for (int i=0; i<3; i++) {
		run_function(pi_par_reduction, "parallel reduction randx4");
	}

	system("pause");
	return 0;
}
