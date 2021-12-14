#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

double generateGaussianNoise(double mu, double sigma)
{
    static bool empty = true;
    static double store;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    constexpr double two_pi = 2.0 * M_PI;

    //initialize the random uniform number generator (runif) in a range 0 to 1
    static std::mt19937 rng(std::random_device{}()); // Standard mersenne_twister_engine seeded with rd()
    static std::uniform_real_distribution<> runif(0.0, 1.0);

    if (empty)
    {
        //create two random numbers, make sure u1 is greater than epsilon
        double u1, u2;
        do
        {
            u1 = runif(rng);
            u2 = runif(rng);
        } while (u1 <= epsilon);
        //compute z0 and z1
        auto mag = sigma * sqrt(-2.0 * log(u1));
        auto z0 = mag * cos(two_pi * u2) + mu;
        auto z1 = mag * sin(two_pi * u2) + mu;

        store = z1;
        empty = false;
        return z0;
    }
    else
    {
        empty = true;
        return store;
    }
    return 0;
}

#include "RingBuf.hpp"
#include "GaussFilter.hpp"

volatile sig_atomic_t done = 0;
void sighandler(int sig)
{
    done = 1;
}

#define BUFFER_SIZE 64

#define print_object(name) \
{ \
    fprintf(stdout, "Object %s: %p\n", #name, &name); \
    fflush(stdout); \
}

int main()
{
    signal(SIGINT, sighandler);

    int endTime = 10000; // ms

    double measure = 10.02;
    double meas_std = 0.1;
    double grad = -1;
    double gradgrad = 0.1;
    double deltaTime = 0.02; // 20 ms
    int order = 5;
    int cutoff_freq = 20;
    GaussFilter<double> ftr(BUFFER_SIZE, cutoff_freq);
    RingBuf<double>
        data_real,    // real data
        data_mes,     // measured data
        data_ftr,     // filtered data
        grad_real,    // real gradient
        grad_mes,     // measured gradient
        grad_ftr,     // gradient of filtered data
        grad_mes_ftr, // gradient of measured data with filter applied
        grad_ftr_ftr; // gradient of filtered data with filter applied

    data_real.Initialize(BUFFER_SIZE);
    data_mes.Initialize(BUFFER_SIZE);
    data_ftr.Initialize(BUFFER_SIZE);
    grad_real.Initialize(BUFFER_SIZE);
    grad_mes.Initialize(BUFFER_SIZE);
    grad_ftr.Initialize(BUFFER_SIZE);
    grad_mes_ftr.Initialize(BUFFER_SIZE);
    grad_ftr_ftr.Initialize(BUFFER_SIZE);
    printf("\n\n\n");
    data_real.push(measure);
    data_mes.push(generateGaussianNoise(measure, meas_std));
    data_ftr.push(data_mes[0]);
    eprintlf("Real: %lf, Meas: %lf, Ftr: %lf", data_real[0], data_mes[0], data_ftr[0]);
    printf("\n\n\n");
    data_ftr[0] = ftr.ApplyFilter(data_ftr);
    // update measurement using gradient
    measure += grad * deltaTime;
    grad += gradgrad * deltaTime;
    // push another set
    data_real.push(measure);
    data_mes.push(generateGaussianNoise(measure, meas_std));
    data_ftr.push(data_mes[0]);
    eprintlf("Real: %lf, Meas: %lf, Ftr: %lf", data_real[0], data_mes[0], data_ftr[0]);
    printf("\n\n\n");
    data_ftr[0] = ftr.ApplyFilter(data_ftr);
    // update measurement using gradient
    measure += grad * deltaTime;
    grad += gradgrad * deltaTime;
    // open plot files
    char fname[128];
    snprintf(fname, sizeof(fname), "data_mes_%d_%d.txt", order, cutoff_freq);
    FILE *plot_mes = fopen(fname, "w");
    snprintf(fname, sizeof(fname), "data_grad_%d_%d.txt", order, cutoff_freq);
    FILE *plot_grad = fopen(fname, "w");
    double time = 2 * deltaTime;
    while (!done)
    {
        int wout = printf("Time: %.2lf s\r", time);
        fflush(stdout);
        // generate and update data
        data_real.push(measure);
        data_mes.push(generateGaussianNoise(measure, meas_std));
        data_ftr.push(data_mes[0]);
        data_ftr[0] = ftr.ApplyFilter(data_ftr);
        grad_real.push(grad);
        grad_mes.push((data_mes[0] - data_mes[2]) / (2 * deltaTime));
        grad_ftr.push((data_ftr[0] - data_ftr[2]) / (2 * deltaTime));
        grad_mes_ftr.push(grad_mes[0]);
        grad_mes_ftr[0] = ftr.ApplyFilter(grad_mes_ftr);
        grad_ftr_ftr.push(grad_ftr[0]);
        grad_ftr_ftr[0] = ftr.ApplyFilter(grad_ftr_ftr);
        // update measurement using gradient
        measure += grad * deltaTime;
        grad += gradgrad * deltaTime;
        // plot
        fprintf(plot_mes, "%lf, %lf, %lf, %lf\n", time, data_real[0], data_mes[0], data_ftr[0]);
        fflush(plot_mes);
        fprintf(plot_grad, "%lf, %lf, %lf, %lf, %lf\n", time, grad_real[0], grad_mes[0], grad_mes_ftr[0], grad_ftr_ftr[0]);
        fflush(plot_grad);
        // wait
        if (endTime > 0 && ((time * 1000) > endTime)) // auto stop
            break;
        // usleep(deltaTime * 1000000);
        time += deltaTime;
        while (wout--)
            printf(" ");
        printf("\r");
    }
    printf("\n\n");
    fclose(plot_mes);
    fclose(plot_grad);
    return 0;
}