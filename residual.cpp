#include <math.h>
#include <omp.h>

double lp_diff_norm(double* a, double* b, int N, double p) {
    double h = 1.0 / (N+1);
    double norm = 0.0;
    double diff;
    int index;
    for(index = 0; index < N*N; index++) {
        diff = fabs(a[index] - b[index]);
        if(p == INFINITY) {
            if(diff > norm) {
                norm = diff;
            }
        } else {
            norm += h * h * pow(diff, p);
        }
    }
    if(p != INFINITY) {
        norm = pow(norm, 1.0 / p);
    }
    return norm;
}


double residual_parallel(int N, double *u, double *f) {
    double resid = 0.0;
    double diff;
    double h = 1.0 / (N + 1);
    
    int i,j, index;
    #pragma omp parallel shared(h) private(i,j,index,diff) reduction(+:resid)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                // First row:
                diff = f[0] + (-4 * u[0] + u[1] + u[N]) / h*h;
                resid += diff * diff;
                for(j = 1; j < N-1; j++) {
                    diff = f[j] + (-4 * u[j] + u[j-1] + u[j+1] + u[j+N]) / h*h;
                    resid += diff * diff;
                }
                diff = f[N-1] + (-4 * u[N-1] + u[N-2] + u[2*N-1]) / h*h;
                resid += diff * diff;
            }

            #pragma omp section
            {
                // Last row:
                index = N*(N-1);
                diff = f[index] + (u[index + 1] + u[index - N] - 4 * u[index])/ h*h;
                resid += diff * diff;
                for(j = 1; j < N-1; j++) {
                    index = N*(N-1) + j;
                    diff = f[index] + (u[index - 1] + u[index + 1]+ u[index - N] - 4 * u[index]) / h*h;
                    resid += diff * diff;
                }
                index = N * N - 1;
                diff = f[index] + (u[index - 1] + u[index - N] - 4 * u[index])/ h*h;
                resid += diff * diff;
            }

        }
        // Interior rows:
        #pragma omp for
        for(i = 1; i < N - 1; i++){
            index = N * i;
            diff = f[index] + (u[index + 1] + u[index + N] + u[index - N] - 4 * u[index]) / h*h;
            resid += diff * diff;
            for(j = 1; j < N - 1; j++) {
                index = N * i + j;
                diff = f[index] + (u[index + 1] + u[index - 1] + u[index + N]  + u[index - N] - 4 * u[index]) / h*h;
                resid += diff * diff;
            }
            index = N * i + (N - 1);
            diff = f[index] + (u[index - 1] + u[index - N] + u[index + N] - 4 * u[index]) / h*h;
            resid += diff * diff;
        }
    }
    return sqrt(resid);
}

double residual_serial(int N, double* u, double *f) {
    double resid = 0.0;
    double diff;
    double h = 1.0 / (N + 1);
   
    int i,j, index;
    // First row:
    diff = f[0] + (-4 * u[0] + u[1] + u[N]) / h*h;
    resid += diff * diff;
    for(j = 1; j < N-1; j++) {
        diff = f[j] + (-4 * u[j] + u[j-1] + u[j+1] + u[j+N]) / h*h;
        resid += diff * diff;
    }
    diff = f[N-1] + (-4 * u[N-1] + u[N-2] + u[2*N-1]) / h*h;
    resid += diff * diff;

    // Interior rows:
    for(i = 1; i < N - 1; i++){
        index = N * i;
        diff = f[index] + (u[index + 1] + u[index + N] + u[index - N] - 4 * u[index]) / h*h;
        resid += diff * diff;
        for(j = 1; j < N - 1; j++) {
            index = N * i + j;
            diff = f[index] + (u[index + 1] + u[index - 1] + u[index + N]  + u[index - N] - 4 * u[index]) / h*h;
            resid += diff * diff;
        }
        index = N * i + (N - 1);
        diff = f[index] + (u[index - 1] + u[index - N] + u[index + N] - 4 * u[index]) / h*h;
        resid += diff * diff;
    }

    // Last row:
    index = N*(N-1);
    diff = f[index] + (u[index + 1] + u[index - N] - 4 * u[index]) / h*h;
    resid += diff * diff;
    for(j = 1; j < N-1; j++) {
        index = N*(N-1) + j;
        diff = f[index] + (u[index - 1] + u[index + 1] + u[index - N] - 4 * u[index]) / h*h;
        resid += diff * diff;
    }
    index = N * N - 1;
    diff = f[index] + (u[index - 1] + u[index - N] - 4 * u[index]) / h*h;
    resid += diff * diff;
    return sqrt(resid);
}

double residual(int N, double* u, double* f) {
    #ifdef _OPENMP
    return residual_parallel(N, u, f);
    #else
    return residual_serial(N, u, f);
    #endif 
}
