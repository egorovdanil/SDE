#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include "mkl.h"

#include "common.h"
#include "StormerVerlet.h"

void GetNoise(float* r, int size, int seed)
{
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT2203, seed);

	vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
		stream, size, r, 0.0, 1.0);
}

void CalculatePropability(float gamma, float omega, float T, float h, float A, float i0, double* mst, double* sd, double* lifetime)
{
	const unsigned short numtasks = 12;

	size_t count = 2400;

	StormerVerletParams params;
	params.A = A;
	params.t_max = T;
	params.h = h;
	params.i0 = i0;
	params.omega = omega;
	params.gamma = gamma;
	params.Cinfigure();

	std::vector<float> P(params.size);
	std::vector<std::vector<float>> P_part(numtasks);
	std::vector<double> lifetime_part(numtasks);
	for (size_t i = 0; i < P_part.size(); i++)
	{
		P_part[i].resize(params.size);
	}

	omp_set_num_threads(numtasks);
#pragma omp parallel shared(numtasks, params)
	{
		short taskid = omp_get_thread_num();
		size_t part_count = count / numtasks;
		if (taskid == 0)
			part_count = part_count + count % numtasks;

		P_part[taskid].resize(params.size, 0.0);

		for (size_t i = 0; i < part_count; i++)
		{
			float* gas = new float[params.size];
			GetNoise(gas, params.size, taskid * (i + 1)); // * i * omega
			//GetStormerVerlet2nd(gas, X, params.i0, params.t_max, params.h, params.omega);
			lifetime_part[taskid] += GetStormerVerlet(gas, P_part[taskid], params, taskid);
			//GetHeun2nd(gas, X, params.i0, params.t_max, params.h, params.omega, taskid + 20);
			delete[] gas;
		}

	}

	for (size_t i = 0; i < numtasks; i++)
	{
		*lifetime += lifetime_part[i];
	}
	*lifetime /= count;

#ifdef LIFE_TIME_BREAK
	return;
#endif

	for (size_t i = 0; i < P.size(); i++)
	{
		P[i] = 0;
	}

#pragma omp for
	{
		for (int i = 0; i < P.size(); ++i)
			for (int p = 0; p < numtasks; ++p)
				P[i] += P_part[p][i];
	}

	for (size_t i = 0; i < P.size(); i++)
	{
		P[i] = P[i] / count;
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
	integral_second *= (params.h * params.h);

	double sqr_integral_mst = (integral_mst * integral_mst) * (params.h * params.h);
	*sd = sqrt(integral_second - sqr_integral_mst);
	*mst = integral_mst / (1 / params.h);
}

void CountMST()
{
	double i0 = 0.9;
	double h = 0.01;
	double A = 0.2;
	double gamma = 0.01;
	double* mst = new double;
	double* sd = new double;
	double* lifetime = new double;

	std::vector<std::pair<double, double>> sds;

	for (double j = 0.2; j <= 0.3; j += 0.1)
	{
		A = j;
		//i0 = 0.95 - j;
		std::cout << "A\t" << j << std::endl;
		std::cout << "i0\t" << i0 << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		auto time_A1 = std::chrono::steady_clock::now();
		for (double k = 0.01; k <= 0.01; k *= 10)
		{
			k = gamma; //
			std::cout << "Gamma\t" << k << "\t\t";
			auto time_gamma1 = std::chrono::steady_clock::now();
			std::string f("C:\\Users\\A\\Desktop\\graphs9\\3tmp12_mst_A_" + std::to_string(A) + "_G_" + std::to_string(k) + ".txt");
			std::string f2("C:\\Users\\A\\Desktop\\graphs9\\3tmp12_sd_A_" + std::to_string(A) + "_G_" + std::to_string(k) + ".txt");
			std::string f3("C:\\Users\\A\\Desktop\\graphs9\\_life_time_A_" + std::to_string(A) + "_G_" + std::to_string(k) + ".txt");
			std::ofstream out(f);
			std::ofstream out2(f2);
			std::ofstream out3(f3);
			double perc = 0;
			std::string perc_str;
			for (double i = 0.0; i <= 1.2; i += 0.002)
			{
				auto time1 = std::chrono::steady_clock::now();
				CalculatePropability(k, i, 1000, h, A, i0, mst, sd, lifetime);
				out << "t = " << i << "\tx = " << *mst << std::endl;
				out2 << "t = " << i << "\tx = " << *sd << std::endl;
				out3 << i << "\t" << *lifetime << std::endl;

				sds.push_back(std::make_pair(*sd, i));

				auto time2 = std::chrono::steady_clock::now();
				perc = i / 1.2 * 100;
				perc_str = std::to_string((int)perc) + "%";

				for (int it_clear = perc_str.size(); it_clear > 0; it_clear -= 1)
					std::cout << "\b";

				std::cout << perc_str;
			}
			out.close();
			out2.close();
			out3.close();
			auto time_gamma2 = std::chrono::steady_clock::now();
			std::cout << "\tTime: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time_gamma2 - time_gamma1).count() << " msec." << std::endl;

			return;
		}

		auto time_A2 = std::chrono::steady_clock::now();
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Totatl time\t" << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time_A2 - time_A1).count() << " msec." << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;
	}

	delete mst;
	delete sd;
}

using namespace std;
void histogram(std::vector<double> mass)
{
	int n = mass.size();
	int max, min;
	double width = 0;

	double ccount = mass.size() / 100;

	min = max = mass[0];
	for (int i = 0; i < n; i++)
	{
		if (mass[i] >= max)
		{
			max = mass[i];
		}
		if (mass[i] <= min)
		{
			min = mass[i];
		}
	}

	vector <int> gist(ccount);
	width = (max - min) / ccount;
	//cout << width;
	for (int i = 0; i < ccount; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if ((mass[j] >= min + i * width) && (mass[j] <= min + (i + 1) * width))
				gist[i]++;
		}
	}


	std::ofstream out("C:\\Users\\A\\Desktop\\graphs3\\___n5.txt");
	for (int i = 0; i < ccount; i++)
	{
		out << "t = " << min + i * width << "\tx = " << gist[i] << std::endl;
		//cout << min + i * width << "| ";
		//for (int j = 0; j < gist[i]; j++)
		//{
		//    //cout << "* ";
		//}
		//cout << endl;
	}
	out.close();
}
