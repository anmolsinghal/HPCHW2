
#ifdef _OPENMP
#include <omp.h>
#define RUN_TYPE "parallel"
#else
#define RUN_TYPE "serial"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "residual.cpp"
#include "utils.h"



void gs_serial(double* u, double* f, int N) {
    int i,j,rb, k;
    double h = 1.0 / (N+1);
    for(rb = 0; rb < 2; rb++) 
    {
        if(rb == 0) 
        {
            k = 0;
            u[k] = (h*h * f[k] + u[k+1] + u[k+N]) / 4;
            k = N*N - 1;
            u[k] = (h*h * f[k] + u[k-1] + u[k-N]) / 4;
            if(N%2 == 1) 
            {
                k = N - 1;
                u[k] = (h*h * f[k] + u[k - 1] + u[k + N]) / 4;
                k = N * (N-1);
                u[k] = (h*h * f[k] + u[k + 1] + u[k - N]) / 4;
            }
        } else 
        { 
            if(N%2 == 0) 
            {
                k = N - 1;
                u[k] = (h*h * f[k] + u[k - 1] + u[k + N]) / 4;
                k = N * (N-1);
                u[k] = (h*h * f[k] + u[k + 1] + u[k - N]) / 4;
            }
        }
        
        for(k = 2-rb; k < N - 1; k+=2) 
        {
            u[k] = (h*h * f[k] + u[k - 1] + u[k + 1]+ u[k + N]) / 4;
        }

        for(k = N*(N-1) + 2 - (rb+N)%2; k < N*N-1; k+=2) 
        {
            u[k] = (h*h * f[k] + u[k - 1] + u[k + 1]+ u[k - N]) / 4;
        }

        for(i = 1; i < N-1; i++) 
        {
            if(i % 2 == rb) 
            {
                k = N*i;
                u[k] = (h*h * f[k] + u[k+1] + u[k+N] + u[k-N]) / 4;
            }
            if((i + N - 1) % 2 == rb) 
            {
                k = N *  (i+1) - 1;
                u[k] = (h*h * f[k] + u[k-1] + u[k+N] + u[k-N]) / 4;
            }
            for(j = 2 - (i+rb)%2; j < N-1; j+=2) {
                k = N*i + j;
                u[k] = (h*h * f[k] + u[k-1] + u[k+1] + u[k+N]\
                            + u[k-N]) / 4;
            }
        }
    }
}

void gs_parallel(double* u, double* f, int N) {
    int i,j,rb, k;
    double h = 1.0 / (N+1);
    for(rb = 0; rb < 2; rb++) 
    {
        #pragma omp parallel shared(u, f, N, rb, h) private(i,j,k)
        {
            #pragma omp sections nowait
            {
                #pragma omp section
                {
                    if(rb == 0) 
                    { 
                        k = 0;
                        u[k] = (h*h * f[k] + u[k+1] + u[k+N]) / 4;
                        k = N*N - 1;
                        u[k] = (h*h * f[k] + u[k-1] + u[k-N]) / 4;
                        if(N%2 == 1) 
                        {
                            k = N - 1;
                            u[k] = (h*h * f[k] + u[k - 1] + u[k + N])\
                                    / 4;
                            k = N * (N-1);
                            u[k] = (h*h * f[k] + u[k + 1] + u[k - N])\
                                     / 4;
                        }
                    } 
                    else 
                    { 
                        if(N%2 == 0) 
                        {
                            k = N - 1;
                            u[k] = (h*h * f[k] + u[k - 1] + u[k + N])/ 4;
                            k = N * (N-1);
                            u[k] = (h*h * f[k] + u[k + 1] + u[k - N])/ 4;
                        }
                    }
                }
                
                #pragma omp section
                {
                    for(k = 2-rb; k < N - 1; k+=2) 
                    {
                        u[k] = (h*h * f[k] + u[k - 1] + u[k + 1]+ u[k + N]) / 4;
                    }
                }

                #pragma omp section
                {
                    for(k = N*(N-1) + 2 - (rb+N)%2; k < N*N-1; k+=2) 
                    {
                        u[k] = (h*h * f[k] + u[k - 1] + u[k + 1] + u[k - N]) / 4;
                    }
                }
            }

            #pragma omp for
            for(i = 1; i < N-1; i++) 
            {
                if(i % 2 == rb) 
                {
                    k = N*i;
                    u[k] = (h*h * f[k] + u[k+1] + u[k+N] + u[k-N])/ 4;
                }
                if((i + N - 1) % 2 == rb) 
                {
                    k = N * (i+1) - 1;
                    u[k] = (h*h * f[k] + u[k-1] + u[k+N] + u[k-N])/ 4;
                }
                for(j = 2 - (i+rb)%2; j < N-1; j+=2) 
                {
                    k = N*i + j;
                    u[k] = (h*h * f[k] + u[k-1] + u[k+1] + u[k+N]+ u[k-N]) / 4;
                }
            }
        }
    }
}

void gs(double* u, double* f, int N) {
    #ifdef _OPENMP
    gs_parallel(u, f, N);
    #else
    gs_serial(u, f, N);
    #endif
}

double run_gs(int iterations, double* f, int N) {
    int iteration;
    double *u, *u_new;
    double res;

    u = (double*)calloc(N * N,  sizeof(double));
    res = residual(N, u, f);    
    for(iteration = 0; iteration < iterations; iteration++) 
    {
        gs(u, f, N);
        res = residual(N, u, f);        
    }    
    free(u_new);
    free(u);
    return res;
}




int main(int argc, char **argv) {
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
    double residual = run_gs(iterations, f, N);
    double time = t.toc();
    printf("Time taken: %.5fs\n", time);
    printf("Final residual: %.5f\n", residual);
    free(f);
    return EXIT_SUCCESS;
}


