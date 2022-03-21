

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "residual.cpp"
#include "utils.h"

#ifdef _OPENMP
#define RUN_TYPE "parallel"
#else
#define RUN_TYPE "serial"
#endif




void jacobi_serial(double *u, double *u_new, double *f, int N) {
    int i,j, k;
    double h = 1.0 / (N + 1);
    u_new[0] = (h*h * f[0] + u[1] + u[N])/4.0;
    for(j = 1; j < N-1; j++) {
        u_new[j] = (h*h * f[j] + u[j-1] + u[j+1] + u[j+N])/4.0;
    }
    u_new[N-1] = (h*h * f[N-1] + u[N-2] + u[2*N-1])/4.0;

    for(i = 1; i < N - 1; i++) {
        u_new[N*i] = (h*h * f[N*i] + u[N * (i - 1)] + u[N * (i + 1)]+ u[N * i + 1])/4.0;
        for(j = 1; j < N - 1; j++) {
            k = N * i + j;
            u_new[k] = (h*h * f[k] + u[k + 1] + u[k - 1] + u[k + N] + u[k - N])/4.0;
        }
        u_new[N*(i + 1) - 1] = (h*h * f[N*(i + 1) - 1] + u[N*(i + 1) - 2]  + u[N*i - 1] + u[N*(i + 2) - 1])/4.0;
    }

    u_new[N*(N-1)] = (h*h * f[N*(N-1)] + u[N * (N-1) + 1] + u[N*(N-2)])/4.0;
    for(j = 1; j < N-1; j++) {
        k = N*(N-1) + j;
        u_new[k] = (h*h * f[k] + u[k-1] + u[k+1] + u[k - N])/4.0;
    }
    u_new[N*N-1] = (h*h * f[N*N-1] + u[N*N-2] + u[N*(N-1)-1])/4.0;
}

void jacobi_parallel(double *u, double *u_new, double* f, int N) {
    int i,j,k;
    double h = 1.0 / (N+1);
    #pragma omp parallel private(i,j,k) shared(u, u_new, h, N, f)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                //First row
                u_new[0] =(h*h * f[0] + u[1] + u[N])/4.0;
                for(j = 1; j < N-1; j++) {
                    u_new[j] = (h*h * f[j] + u[j-1] + u[j+1] + u[j+N])/4.0;
                }
                u_new[N-1] =(h*h * f[N-1] + u[N-2] + u[2*N-1])/4.0;
            }

            #pragma omp section
            {
                //Last row
                k = N * (N-1);
                u_new[k] = (h*h * f[k] + u[k + 1] + u[k - N])/4.0;
                for(j = 1; j < N-1; j++) {
                    k = N*(N-1) + j;
                    u_new[k] = (h*h * f[k] + u[k - 1]+ u[k + 1] + u[k - N])/4.0;
                }
                k = N*N - 1;
                u_new[k] = (h*h * f[k] + u[k - 1]+ u[k - N])/4.0;
            }
        }

        #pragma omp for
        for(i=1; i < N-1; i++) {
            k = N*i;
            u_new[k] = (h*h * f[k] + u[k - N] + u[k + N]+ u[k + 1])/4.0;
            for(j=1; j<N-1; j++) {
                k = N*i + j;
                u_new[k] = (h*h * f[k] + u[k-1]+ u[k+1] + u[k-N] + u[k+N])/4.0;
            }
            k = N * (i + 1) - 1;
            u_new[k] = (h*h * f[k] + u[k - 1] + u[k - N]+ u[k + N])/4.0;
        }
    }
}

void jacobi(double *u, double *u_new, double *f, int N) {
    #ifdef _OPENMP
    jacobi_parallel(u, u_new, f, N);
    #else
    jacobi_serial(u, u_new, f, N);
    #endif
}

double jacobi(int N, int iterations, double *f) {
    int iteration;
    double *u, *u_new;
    double res;

    u = (double*)calloc(N * N,  sizeof(double));
    u_new = (double*)malloc(N * N * sizeof(double));

    res = residual(N, u, f);
    
    for(iteration = 0; iteration < iterations; iteration++) {
        jacobi(u, u_new, f, N);
        double *swp_tmp = u;
        u = u_new;
        u_new = swp_tmp;
        res = residual(N, u, f);
        
    }
    
    free(u_new);
    free(u);
    return res;
}


int main(int argc, char **argv){
    
    int i,j;
    int N = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    int threads = atoi(argv[3]);
    omp_set_num_threads(threads);
    double *f = (double*)malloc(N * N * sizeof(double));
    double elapsed;
    for(i = 0; i < N*N; i++) {
        f[i] = 1;
    }
    Timer t;
    t.tic();
    double residual = jacobi(N, iterations, f);
    double time = t.toc();
    printf("Time taken: %.5fs\n", time);
    printf("Final residual: %.5f\n", residual);
    free(f);
    return EXIT_SUCCESS;
}

