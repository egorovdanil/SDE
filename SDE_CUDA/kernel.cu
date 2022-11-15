#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <fstream>
#include <map>

#include <math.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

# define M_PI 3.14159265358979323846

struct KernelSize
{
    size_t threadsPerBlock;
    size_t blockCount;
    KernelSize() {}
    KernelSize(size_t threadsPerBlock_, size_t blockCount_)
        : threadsPerBlock(threadsPerBlock_), blockCount(blockCount_)
    {
    }
};

std::map<size_t, KernelSize> g_kernel_size_per_threads
{
    {1, KernelSize(1, 1)},
    {2, KernelSize(2, 1)},
    {4, KernelSize(4, 1)},
    {8, KernelSize(8, 1)},
    {12, KernelSize(6, 2)},
    {16, KernelSize(8, 2)},
    {24, KernelSize(12, 2)},
    {32, KernelSize(16, 2)},
    {48, KernelSize(24, 2)},
    {64, KernelSize(16, 4)},
    {256, KernelSize(32, 8)},
    {512, KernelSize(32, 16)},
    {1024, KernelSize(64, 16)},
    {2048, KernelSize(64, 32)},
};

struct RealParams
{
    double T = 100;
    double Ic = 0.0851;
    double Ib = 0.0415;
    double Cp = 0.036;
    double Rn = 3400;
    double A = 0;
    double Fq = 0;

    double Ib_step = 0.001;
};

struct Params
{
    double omega;
    double omega_max;
    double omega_step;

    double amplitude;
    double gamma;

    double alpha;
    double i0;
    double t;
    double t_max;
    double h;

    double b;
    double a;

    double noise_intensity;
    double c_h_half;
    double c_bh_half;
    double c_bh;
    double c_bh_squar;

    double h_barrier;
    double l_barrier;

    double h_barrier_v;
    double l_barrier_v;

    double Wp;
    double seconds;

    size_t size;
    RealParams real;

    Params()
    {
        omega = 0.0;
        omega_max = 1.2;
        omega_step = 0.01;

        amplitude = 0.3;
        gamma = 0.01;
        alpha = 0.1;
        i0 = 0.5;
        t = 0;
        t_max = 1.1326e+10;
        h = 1;

        seconds = 0.1;

        Configure();
    };

    void ConfigureFromReal()
    {
        const long double c_s2eh = 55122907194.91327; // sqrt(2 * e / h)* sqrt(CI / CC)
        const long double c_1s2eh = 18.141278297678028; // c_s2eh / (2 * e * CI / h)
        const long double c_2ekh = 4.195150167862804e-05; // 2 * e * k / h

        Wp = c_s2eh * sqrt(real.Ic / real.Cp) / (2.0 * M_PI);

        i0 = real.Ib / real.Ic;
        amplitude = real.A * real.Ic;
        omega = real.Fq / Wp;
        alpha = c_1s2eh / (real.Rn * sqrt(real.Ic * real.Cp));
        gamma = c_2ekh * real.T / real.Ic;

        t_max = Wp * seconds;

        Configure();
    }

    void Configure()
    {
        h_barrier = M_PI + asin(amplitude - i0);
        l_barrier = -h_barrier;

        h_barrier_v = 0.5 * i0 / alpha;
        l_barrier_v = -h_barrier_v;

        b = 1.0 / (1.0 + alpha * (h * 0.5));
        a = b / (1.0 + alpha * (h * 0.5));

        noise_intensity = sqrt(2.0 * alpha * gamma * h);
        c_h_half = h * 0.5;
        c_bh_half = b * c_h_half;
        c_bh = h * b;
        c_bh_squar = c_bh * c_h_half;
        if (c_bh_half != (b * (h * 0.5f))) std::cout << c_bh_half << " " << b * (h * 0.5) << "Err\n" << std::endl;

        size = t_max / h;
    }

    int IncrementOmega()
    {
        omega += omega_step;
        if (omega <= omega_max)
        {
            Configure();
            return 0;
        }
        else
        {
            return -1;
        }
    }
};

__device__ void storemer()
{
    float x = 2.0;
    __sinf(x);
}

__global__ void calc_kernel(Params params, double* phase, double* voltage)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState;
    curand_init(params.omega, id, 0, &localState);

    double lifetime;
    lifetime = 0.0;
    bool end_phase;
    end_phase = false;
    bool end_voltage;
    end_voltage = false;
    double fi_current;
    fi_current = asin(params.i0);
    double sin_fi;
    sin_fi = 0;
    double fi_next;
    fi_next = 0;
    double v_current;
    v_current = 0;
    double v_next;
    v_next = 0;
    double noise;
    noise = 0;

    double t;
    t = 0;

    double res_phase;
    res_phase = params.t_max;
    double res_voltage;
    res_voltage = params.t_max;

    float2 gas;
    double it;

    for (size_t i = 0; i < params.size / 2; ++i)
    {
        if (!end_phase)
        {
            if (!(fi_current <= params.h_barrier && fi_current >= params.l_barrier))
            {
                res_phase = t;
                end_phase = true;
                break;
            }
        }

        if (!end_voltage)
        {
            if (!(v_current <= params.h_barrier_v && v_current >= params.l_barrier_v))
            {
                res_voltage = t;
                end_voltage = true;
            }
        }

        gas = curand_normal2(&localState);

        ////////////////////////////

        it = params.i0 + params.amplitude * __sinf((params.omega * t));

        noise = params.noise_intensity * gas.x;
        sin_fi = __sinf(fi_current);

        fi_next = fi_current + params.c_bh * v_current - params.c_bh_squar * (sin_fi - it) + params.c_bh_half * noise;
        v_next = params.a * v_current - params.c_h_half * (params.a * sin_fi + __sinf(fi_next)) + it * params.c_bh + params.b * noise;

        fi_current = fi_next;
        v_current = v_next;

        t += params.h;

        ////////////////////////////

        it = params.i0 + params.amplitude * __sinf((params.omega * t));

        noise = params.noise_intensity * gas.y;
        sin_fi = __sinf(fi_current);

        fi_next = fi_current + params.c_bh * v_current - params.c_bh_squar * (sin_fi - it) + params.c_bh_half * noise;
        v_next = params.a * v_current - params.c_h_half * (params.a * sin_fi + __sinf(fi_next)) + it * params.c_bh + params.b * noise;

        fi_current = fi_next;
        v_current = v_next;

        t += params.h;
    }

    phase[id] = res_phase;
    voltage[id] = res_voltage;
}

struct Result
{
    double mst;
    double sd;
    double mst_v;
    double sd_v;
};
// totalThreads, streams_count
__global__ void calc_probability_kernel(double* timings, size_t count, size_t shift, Result* res, short type)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    double mst = 0;
    double sd = 0;
    for (size_t i = id * count; i < count + id * count; ++i)
    {
        mst += timings[i];
    }
    mst /= count;

    for (size_t i = id * count; i < count + id * count; ++i)
    {
        sd += (timings[i] - mst) * (timings[i] - mst);
    }
    sd = sqrt(sd / count);

    if (type == 0) {
        res[id].mst = mst;
        res[id].sd = sd;
    }
    else if (type == 1) {
        res[id].mst_v = mst;
        res[id].sd_v = sd;
    }
}

void calc_probability(float* phase_timings, float* voltage_timings, Params params_, size_t count, Result* res)
{
    auto calc = [](const float* const timings, const Params& params_, const size_t& count)
    {
        std::vector<float> P(params_.size, 0.0);
        for (size_t i = 0; i < count; i++)
        {
            size_t t_index = 0;
            while (timings[i] > t_index * params_.h && t_index < params_.size)
            {
                P[t_index++] += 1.0;
            }
        }

        for (size_t i = 0; i < P.size(); i++)
        {
            P[i] /= count;
        }

        double t = 1 * 0.5;
        double integral_mst = 0;
        double integral_second = 0;
        for (size_t i = 0; i < P.size(); i++)
        {
            integral_mst += P[i];
            integral_second += t * P[i];
            t += 1;
        }
        integral_second *= 2.0;
        integral_second *= (params_.h * params_.h);
        double sqr_integral_mst = (integral_mst * integral_mst) * (params_.h * params_.h);
        float sd = sqrt(integral_second - sqr_integral_mst);
        float mst = integral_mst * params_.h;
        return std::make_pair(mst, sd);
    };
    auto calc_v2 = [](const float* const timings, const Params& params_, const size_t& count)
    {
        double mst = 0;
        double sd = 0;
        for (size_t i = 0; i < count; ++i)
        {
            mst += timings[i];
        }
        mst /= count;

        for (size_t i = 0; i < count; ++i)
        {
            sd += (timings[i] - mst) * (timings[i] - mst);
        }
        sd = sqrt(sd / count);

        return std::make_pair(mst, sd);
    };

    auto res_pahse = calc(phase_timings, params_, count);
    res->mst = res_pahse.first;
    res->sd = res_pahse.second;
    auto res_voltage = calc(voltage_timings, params_, count);
    res->mst_v = res_pahse.first;
    res->sd_v = res_pahse.second;
}

struct Task
{
    Params params;
    Result* host_results;
    size_t size;
    std::vector<std::thread> threads;
    std::vector<double*> host_phase_timings;
    std::vector<double*> host_voltage_timings;

    Task(Params params_)
        : params(params_)
    {
        params.Configure();
        pipeline(params);
    }

    int pipeline(Params initial_params_)
    {
        auto time1 = std::chrono::steady_clock::now();

        const unsigned int threadsPerBlock = 16;
        const unsigned int blockCount = 16;
        size_t totalThreads = threadsPerBlock * blockCount;

        RealParams real;
        std::vector<Params> params;
        initial_params_.real.Ib = real.Ib;
        for (size_t i = 0; i < 12; i++)
        {
            initial_params_.ConfigureFromReal();
            params.push_back(initial_params_);
            initial_params_.real.Ib += initial_params_.real.Ib_step;
        }

        size_t streams_count = params.size();
        size = streams_count;
        std::vector<cudaStream_t> streams(streams_count + 1);

        host_phase_timings.resize(streams_count);
        host_voltage_timings.resize(streams_count);

        std::vector<double*> dev_phase_timings(streams_count);
        std::vector<double*> dev_voltage_timings(streams_count);

        for (size_t i = 0; i < streams_count; ++i)
        {
            CUDA_CALL(cudaStreamCreate(&streams[i]));
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (int i = 0; i < streams_count; ++i)
        {
            CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&dev_phase_timings[i]), totalThreads * sizeof(double), streams[i]));
            CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&dev_voltage_timings[i]), totalThreads * sizeof(double), streams[i]));
            CUDA_CALL(cudaMemsetAsync(dev_phase_timings[i], 0, totalThreads * sizeof(double), streams[i]));
            CUDA_CALL(cudaMemsetAsync(dev_voltage_timings[i], 0, totalThreads * sizeof(double), streams[i]));
        }

        CUDA_CALL(cudaDeviceSynchronize());

        auto time2 = std::chrono::steady_clock::now();
        std::cout << "Init time: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() << std::endl;
        auto time21 = std::chrono::steady_clock::now();


        for (size_t i = 0; i < streams_count; ++i)
        {
            calc_kernel << < threadsPerBlock, blockCount, 1, streams[i] >> > (params[i], dev_phase_timings[i], dev_voltage_timings[i]);
        }

        CUDA_CALL(cudaDeviceSynchronize());

        auto time22 = std::chrono::steady_clock::now();
        std::cout << "Kernel time: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time22 - time21).count() << std::endl;
        auto time31 = std::chrono::steady_clock::now();

        for (size_t i = 0; i < streams_count; ++i)
        {
            host_phase_timings[i] = (double*)calloc(totalThreads, sizeof(double));
            host_voltage_timings[i] = (double*)calloc(totalThreads, sizeof(double));
            CUDA_CALL(cudaMemcpyAsync(host_phase_timings[i], dev_phase_timings[i], totalThreads * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CALL(cudaMemcpyAsync(host_voltage_timings[i], dev_voltage_timings[i], totalThreads * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
        }

        CUDA_CALL(cudaDeviceSynchronize());

        Result* dev_results;
        double* dev_phase_timings_linear;
        double* dev_voltage_timings_linear;
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_results), streams_count * sizeof(Result)));
        CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&dev_phase_timings_linear), totalThreads * streams_count * sizeof(double), streams[0]));
        CUDA_CALL(cudaMallocAsync(reinterpret_cast<void**>(&dev_voltage_timings_linear), totalThreads * streams_count * sizeof(double), streams[1]));

        CUDA_CALL(cudaDeviceSynchronize());

        for (size_t i = 0; i < streams_count; ++i)
        {
            CUDA_CALL(cudaMemcpyAsync(dev_phase_timings_linear + (i * totalThreads), dev_phase_timings[i], totalThreads * sizeof(double), cudaMemcpyDeviceToDevice, streams[0]));
            CUDA_CALL(cudaMemcpyAsync(dev_voltage_timings_linear + (i * totalThreads), dev_voltage_timings[i], totalThreads * sizeof(double), cudaMemcpyDeviceToDevice, streams[1]));
        }

        CUDA_CALL(cudaDeviceSynchronize());

        calc_probability_kernel << <
            g_kernel_size_per_threads[streams_count].threadsPerBlock,
            g_kernel_size_per_threads[streams_count].blockCount,
            0,
            streams[0]
            >> >
            (dev_phase_timings_linear, totalThreads, streams_count, dev_results, 0);

        calc_probability_kernel << <
            g_kernel_size_per_threads[streams_count].threadsPerBlock,
            g_kernel_size_per_threads[streams_count].blockCount,
            0,
            streams[1]
            >> >
            (dev_voltage_timings_linear, totalThreads, streams_count, dev_results, 1);

        CUDA_CALL(cudaDeviceSynchronize());

        // copy res
        host_results = (Result*)calloc(streams_count, sizeof(Result));
        CUDA_CALL(cudaMemcpy(host_results, dev_results, streams_count * sizeof(Result), cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaDeviceSynchronize());

        for (size_t i = 0; i < streams_count; ++i)
        {
            CUDA_CALL(cudaFreeAsync(dev_phase_timings[i], streams[i]));
            CUDA_CALL(cudaFreeAsync(dev_voltage_timings[i], streams[i]));
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (size_t i = 0; i < streams_count; ++i)
        {
            CUDA_CALL(cudaStreamDestroy(streams[i]));
        }

        for (size_t i = 0; i < host_phase_timings.size(); ++i)
        {
            free(host_phase_timings[i]);
            free(host_voltage_timings[i]);
        }

        auto time32 = std::chrono::steady_clock::now();
        std::cout << "Post proc time: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time32 - time31).count() << std::endl;

        std::cout << "OK!" << std::endl;


        return EXIT_SUCCESS;
    }
};

int main(int argc, char* argv[])
{
    auto time1 = std::chrono::steady_clock::now();
    {
        std::vector<Task> tasks;

        RealParams real;
        Params initial_params;
        std::vector<double> Hz = { (double)real.Fq };

        for (auto x : Hz) {
            initial_params.real.Fq = x;
            tasks.push_back(initial_params);
        }
        initial_params.ConfigureFromReal();

        std::string path = "C:\\Fit2\\";

        std::string file_name = "phase";
        std::string file_name2 = "voltage";

        for (size_t i = 0; i < tasks.size(); ++i)
        {
            for (size_t c = 0; c < 1; c++)
            {
                std::string num = "Ic_" + std::to_string(real.Ic);
                real.Ic += 0.0002;

                std::ofstream out_mst(path + num + "mst_" + file_name + std::to_string(Hz[i]) + ".txt");
                std::ofstream out_sd(path + num + "sd_" + file_name + std::to_string(Hz[i]) + ".txt");

                std::ofstream out_mst2(path + num + "mst_" + file_name2 + std::to_string(Hz[i]) + ".txt");
                std::ofstream out_sd2(path + num + "sd_" + file_name2 + std::to_string(Hz[i]) + ".txt");

                initial_params.real.Ib = real.Ib;
                for (size_t j = 0; j < 12; j++)
                {
                    std::cout << tasks[i].host_results[j].mst << std::endl;
                    out_mst << initial_params.real.Ib << "\t" << tasks[i].host_results[j].mst / initial_params.Wp << std::endl;
                    initial_params.real.Ib += initial_params.real.Ib_step;
                }

                initial_params.real.Ib = real.Ib;
                for (size_t j = 0; j < 12; j++)
                {
                    out_sd << initial_params.real.Ib << "\t" << tasks[i].host_results[j].sd / initial_params.Wp << std::endl;
                    initial_params.real.Ib += initial_params.real.Ib_step;
                }
            }
        }
    }

    auto time2 = std::chrono::steady_clock::now();
    std::cout << "Time: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() << std::endl;
    return EXIT_SUCCESS;
}
