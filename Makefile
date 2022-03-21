all: jacobi gaussSeidel val_test01 val_test02 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 MMult1

jacobi: jacobi2D-omp.cpp residual.cpp utils.h
	g++ -std=c++11 -O3 -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp

gaussSeidel: gs2D-omp.cpp residual.cpp utils.h
	g++ -std=c++11 -O3 -fopenmp gs2D-omp.cpp  -o gs2D-omp

val_test01: val_test01_solved.cpp
	g++ -std=c++11 val_test01_solved.cpp -o val_test01_solved

val_test02: val_test01_solved.cpp
	g++ -std=c++11 val_test02_solved.cpp -o val_test02_solved

omp_solved2: omp_bug2.c
	g++ -std=c++11 -O3 -fopenmp omp_bug2.c -o omp_solved2

omp_solved3: omp_bug3.c
	g++ -std=c++11 -O3 -fopenmp omp_bug3.c -o omp_solved2

omp_solved4: omp_bug4.c
	g++ -std=c++11 -O3 -fopenmp omp_bug4.c -o omp_solved2

omp_solved5: omp_bug5.c
	g++ -std=c++11 -O3 -fopenmp omp_bug5.c -o omp_solved2

omp_solved6: omp_bug6.c
	g++ -std=c++11 -O3 -fopenmp omp_bug6.c -o omp_solved2

MMult1: MMult1.cpp
	g++ -std=c++11 -O3 -fopenmp MMult1.cpp -o MMult1
clean:
	rm -rf *.out
	rm jacobi2D-omp
	rm gs2D-omp
	rm MMult1
	rm omp_solved2
	rm omp_solved3
	rm omp_solved4
	rm omp_solved5
	rm omp_solved6
	rm val_test01_solved
	rm val_test02_solved

