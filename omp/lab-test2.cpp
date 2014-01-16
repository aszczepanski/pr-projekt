#include <omp.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <Windows.h>

using namespace std;

#define TAB_SIZE ((size_t) 1<<28)
#define X_SIZE ((size_t) 1e3)
#define Y_SIZE ((size_t) 1e5)

typedef int el_type;

el_type tab[TAB_SIZE];
el_type tab2d[Y_SIZE][X_SIZE];

char empty[900000];

HANDLE thread_uchwyt=GetCurrentThread();

double start;

void print_elapsed_time()
{
        double elapsed ;
        double resolution ;

        // wyznaczenie i zapisanie czasu przetwarzania
        elapsed = (double) clock() / CLK_TCK ;
        resolution = 1.0 / CLK_TCK ;
		printf("Czas: %8.4f sec \n",
                elapsed-start) ;
}

int clear_cache() {
	int xxx;
	for (size_t i = 0; i<900000; i++) {
		xxx = empty[i];
	}
	return xxx;
}

// initialize tab
void init_tab() {
	int i;
#pragma omp parallel for shared(tab) private(i)
	for (i=0; i<TAB_SIZE; i++) {
		tab[i] = (el_type)(i)/1000;
	}
}

void init_tab2d() {
	int i;
#pragma omp parallel for shared(tab) private(i)
	for (i=0; i<X_SIZE*Y_SIZE; i++) {
		tab2d[i/X_SIZE][i%X_SIZE] = (el_type)(i)/1000;
	}
}

// sequential sum
el_type sums() {
	el_type sum = 0;

	for (size_t i=0; i<TAB_SIZE; i++) {
		size_t& k = i;
		sum += tab[k];
	}

	return sum;
}


// sekcyjnosc
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


// omp sum
el_type sum_omp_reduction2() {
	int j;
	el_type sum = 0;

#pragma omp parallel for default(none) shared(tab) private(j) reduction(+:sum) // schedule(dynamic,8192)
for (j = 0; j < 32; j++) {
		for (size_t i=0; i<TAB_SIZE; i+=32) {
			int k = i + j;
			sum += tab[k];
		}
	}

return sum;
}

// omp sum for secetion
el_type sum_omp_reduction() {
	int i;
	el_type sum = 0;

#pragma omp parallel for default(none) shared(tab) private(i) reduction(+:sum) // schedule(dynamic,8192)
for (i=0; i<TAB_SIZE; i++) {
		sum += tab[i];
	}

return sum;
}


int main(int argc, char* argv[]) {

	cout << "element size: " << sizeof(el_type) << endl;

	omp_set_num_threads(4);

#pragma omp parallel
	{

#pragma omp single
		{
			cout << "num threads: " <<  omp_get_num_threads() << endl;
		}
		const int liczba_procesorow = 4;
		int th_id=omp_get_thread_num();
		//otrzymanie własnego identyfikatora
		DWORD_PTR mask = (1 << (th_id % liczba_procesorow ));
		//określenie maski dla przetwarzania wątku wyłącznie na jednym procesorze\
		przydzielanym modulo
		DWORD_PTR result = SetThreadAffinityMask(thread_uchwyt, mask);
		//przekazanie do systemu operacyjnego maski powinowactwa
		if (result==0) printf("blad SetThreadAffnityMask \n");
		else {
			printf("maska poprzednia dla watku %d : %d\n",th_id,result);
			printf("maska nowa dla watku %d : %d\n",th_id,SetThreadAffinityMask(
				thread_uchwyt, mask ));
		}
		//sprawdzenie poprawności ustlenia maski powinowactwa
	}

	Sleep(1000);

	start = (double) clock() / CLK_TCK ;
	init_tab();
	cout << "init: ";
	print_elapsed_time() ;

	Sleep(1000);

	#pragma omp parallel
	{
	clear_cache();
	}
	cout << "clear done\n";
	
	Sleep(1000);

	start = (double) clock() / CLK_TCK ;
	cout << sum_omp_reduction() << endl;
	print_elapsed_time() ;

	Sleep(1000);

	#pragma omp parallel
	{
	clear_cache();
	}
	cout << "clear done\n";
	
	Sleep(1000);

	start = (double) clock() / CLK_TCK ;
	cout << sum_omp_reduction2() << endl;
	print_elapsed_time() ;

	Sleep(1000);

	#pragma omp parallel
	{
	clear_cache();
	}
	cout << "clear done\n";
	
	Sleep(1000);

	start = (double) clock() / CLK_TCK ;
	cout << sums() << endl;
	print_elapsed_time() ;

	Sleep(1000);

	#pragma omp parallel
	{
	clear_cache();
	}
	cout << "clear done\n";
	
	Sleep(1000);

	start = (double) clock() / CLK_TCK ;
	cout << sums2() << endl;
	print_elapsed_time() ;

	system("pause");
	return 0;
}
