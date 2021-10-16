#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>
#include <cmath> 
#define PI 3.14159265

#include "mkl.h"

void GetHeun2nd(std::vector<double> gas, std::vector<double>& x, double _i0 = 0.7, double _T = 20, double _h = 0.01, double _omega = 0.62)
{
	double A = 0.7;
	double omega = _omega;
	double gamma = 0.001;
	double alpha = 0.1;
	double i0 = _i0;

	double t = 0;
	double t_max = _T;
	double h = _h;

	double fi_current = asin(i0);
	double fi_next = 0;
	double v_current = 0;
	double v_next = 0;

	double X_inter;
	double V_inter;

	size_t size = x.size();
	for (size_t i = 0; i < size; i++)
	{
		double it = i0 + A * sin(omega * t);
		double noise = sqrt(2.0 * alpha * gamma * h) * gas[i];

		X_inter = fi_current + h * v_current;
		V_inter = v_current - h * (alpha * v_current + sin(fi_current) - it) + noise;

		fi_next = fi_current + h * 0.5 * (V_inter + v_current);
		v_next = v_current - h * 0.5 * (alpha * (V_inter + v_current) + sin(fi_current) + sin(X_inter)) + it * h + noise;

		t += h;

		x[i] = fi_current;
		//out << "t = " << t << "\tx = " << fi_current << std::endl;

		fi_current = fi_next;
		v_current = v_next;
	}
	//out.close();
}

struct StormerVerletParams
{
	double A;
	double omega;
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

	size_t size;
	std::vector<double> it;

	StormerVerletParams()
	{
		A = 0.6;
		omega = 0.5;
		gamma = 0.007;
		alpha = 0.1;
		i0 = 0.5;
		t = 0;
		t_max = 20;
		h = 0.01;
		Cinfigure();
	};
	void Cinfigure()
	{
		CalculateCoefs();
		CalculateItByTime();
	}
	void CalculateCoefs()
	{
		b = 1.0 / (1.0 + alpha * (h * 0.5));
		a = b / (1.0 + alpha * (h * 0.5));

		noise_intensity = sqrt(2.0 * alpha * gamma * h);
		c_h_half = h * 0.5;
		c_bh_half = b * c_h_half;
		c_bh = h * b;
		c_bh_squar = c_bh * c_h_half;
		if (c_bh_half != (b * (h * 0.5))) std::cout << "Err\n" << std::endl;

		size = t_max / h;
		it.resize(size);
	};

	void CalculateItByTime()
	{
		double t = 0;
		for (size_t i = 0; i < size; ++i)
		{
			it[i] = i0 + A * sin(omega * t);
			t += h;
		}
	}

	void ResetStartParams()
	{
	}
};

std::vector<double> GetNoise(int size, int seed)
{
	float* r = new float[size];
	VSLStreamStatePtr stream;

	// initialize RNG on MIC
#pragma offload target(mic) in(seed) nocopy(stream)
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);

#pragma offload target(mic) \
	in(n) out(r) nocopy(stream)
	{
		vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
			stream, size, r, 0.0, 1.0);
	}

	std::vector<double> gas;
	for (size_t i = 0; i < size; i++)
	{
		gas.push_back((double)r[i]);
	}

	delete r;
	return gas;
}

void GetStormerVerlet(std::vector<double> gas, std::vector<double>& X, StormerVerletParams& params)
{
	double fi_current = asin(params.i0);
	double sin_fi = 0;
	double fi_next = 0;
	double v_current = 0;
	double v_next = 0;
	double noise = 0;

	for (size_t i = 0; i < params.size; ++i)
	{
		X[i] = fi_current;

		noise = params.noise_intensity * gas[i];
		sin_fi = sin(fi_current);

		fi_next = fi_current + params.c_bh * v_current - params.c_bh_squar * (sin_fi - params.it[i]) + params.c_bh_half * noise;
		v_next = params.a * v_current - params.c_h_half * (params.a * sin_fi + sin(fi_next)) + params.it[i] * params.c_bh + params.b * noise;

		fi_current = fi_next;
		v_current = v_next;
	}
}

void MergeP(std::vector<double>& P1, std::vector<double>& P2)
{
	for (size_t i = 0; i < P1.size(); i++)
	{
		P1[i] += P2[i];
	}
}

void CalculatePropability(double gamma, double omega, double T, double h, double A, double i0, double* mst, double* sd)
{
	static const unsigned numtasks = 12;

	size_t count = 2400;

	std::vector<double> gas_parts;
	std::vector<double> P_parts(numtasks);
	
	StormerVerletParams params;
	params.A = A;
	params.t_max = T;
	params.t = 0;
	params.h = h;
	params.i0 = i0;
	params.omega = omega;
	params.gamma = gamma;
	params.Cinfigure();

	static double h_barrier = PI + asin(A - i0);
	static double l_barrier = -h_barrier;
	
	//std::vector<double> gas(params.size);
	//gas = GetNoise(params.size, rand());
	//std::vector<double> P_s(params.size);
	//{
	//	std::vector<double> X(params.size);

	//	for (size_t i = 0; i < P_s.size(); i++)
	//	{
	//		P_s[i] = 0;
	//	}

	//	for (size_t i = 0; i < count; i++)
	//	{
	//		GetStormerVerlet(gas, X, params);
	//		for (size_t j = 0; j < X.size(); j++)
	//		{
	//			if (X[j] <= h_barrier && X[j] >= l_barrier) P_s[j] += 1;
	//		}
	//	}
	//	
	//	for (size_t i = 0; i < P_s.size(); i++)
	//	{
	//		P_s[i] = P_s[i] / count;
	//	}
	//}

	std::vector<double> P(params.size);
	std::vector<std::vector<double>> P_part(numtasks);
	for (size_t i = 0; i < P_part.size(); i++)
	{
		P_part[i].resize(params.size);
	}

#pragma omp parallel shared(numtasks, params)
{
	short taskid = omp_get_thread_num();
	size_t part_count = count / numtasks;
	if (taskid == 0) part_count = part_count + count % numtasks;

	std::vector<double> X(params.size);
	std::vector<double> Y(params.size);
	std::vector<double> gas(params.size);
	//std::vector<double> gas2(params.size);
	P_part[taskid].resize(params.size);
	for (size_t i = 0; i < P_part[taskid].size(); i++)
	{
		P_part[taskid][i] = 0;
	}

	//std::cout << "id: " << taskid << " part_count = " << part_count << std::endl;

	for (size_t i = 0; i < part_count; i++)
	{
		gas = GetNoise(params.size, taskid * i * omega);
		//gas2 = GetNoise(params.size, taskid * i * omega);
		GetStormerVerlet(gas, X, params);
		//GetHeun2nd(gas, X, params.i0, params.t_max, params.h, params.omega);
		//std::ofstream out("C:\\Users\\A\\Desktop\\graphs2\\____00H" + std::to_string(i) + ".txt");
		//std::ofstream out2("C:\\Users\\A\\Desktop\\graphs2\\____00V" + std::to_string(i) + ".txt");
		for (size_t j = 0; j < X.size(); j++)
		{
			if (X[j] <= h_barrier && X[j] >= l_barrier) P_part[taskid][j] += 1;
			//out << "t = " << j * params.h << "\tx = " << X[j] << std::endl;
			//out2 << "t = " << j * params.h << "\tx = " << Y[j] << std::endl;
		}
	}

	//if (taskid % 2 == 0)
	//{
	//	short part_shift = 1;
	//	short related_parts = numtasks / 2;
	//	while (related_parts % 2 == 0)
	//	{
	//		//std::cout << "id: " << taskid << " merging " << taskid << "and" << taskid + part_shift << std::endl;
	//		MergeP(P_part[taskid], P_part[taskid + part_shift]);
	//		part_shift *= 2;
	//		related_parts /= 2;
	//	}
	//}
}

	//std::cout << "id: " << taskid << " merging " << taskid << "and" << taskid + part_shift << std::endl;
	MergeP(P_part[0], P_part[1]);
	MergeP(P_part[0], P_part[2]);
	MergeP(P_part[0], P_part[3]);
	MergeP(P_part[0], P_part[4]);
	MergeP(P_part[0], P_part[5]);
	MergeP(P_part[0], P_part[6]);
	MergeP(P_part[0], P_part[7]);
	MergeP(P_part[0], P_part[8]);
	MergeP(P_part[0], P_part[9]);
	MergeP(P_part[0], P_part[10]);
	MergeP(P_part[0], P_part[11]);

	for (size_t i = 0; i < P.size(); i++)
	{
		P[i] = P_part[0][i] / count;
	}

	//for (size_t i = 0; i < P.size(); i++)
	//{
	//	if (P[i] != P_s[i])
	//	{
	//		std::cout << "err: " << P[i] << " != " << P_s[i] << std::endl;
	//	} 
	//}
	double t = 0;
	double integral_mst = 0;
	double integral_second = 0;
	size_t i = 0;
	for (double t = 0; t <= params.h * 0.5; t += h) {}

	for (; i < P.size(); i++)
	{
		integral_mst += P[i];
		integral_second += t * P[i];
		t += params.h;
	}
	integral_mst *= params.h;
	integral_second *= params.h;
	integral_second *= 2.0;

	//integral_mst *= w_const;
	//integral_second *= w_const;
	//integral_mst -= 0.01;

	//integral_second += params.t_max;
	//std::cout << P[0] << std::endl;
	//integral_mst = (params.t_max * P[P.size() - 1] + params.h * P[1]) + integral_mst;
	//integral_second = (params.t_max * params.t_max * P[P.size() - 1] + params.h * params.h * P[1]) + integral_second;

	//double t = 0;
	//double derivative = 0;
	//double derivative_2 = 0;
	//double integral_mst = 0;
	//double integral_second = 0;

	//double w_const = 1 / (P[P.size() - 1] - P[0]);
	//derivative = (P[P.size() - 1] - P[P.size() - 3]) / (2 * params.h);
	//derivative_2 = (P[2] - P[0]) / (2 * params.h);

	//integral_mst = params.h * ((params.t_max * (w_const * derivative) + t * (w_const * derivative_2)) / 2.0);
	//integral_second = params.h * ((params.t_max * params.t_max * (w_const * derivative) + t * t * (w_const * derivative_2)) / 2.0);
	//t += params.h;

	//for (size_t i = 1; i < P.size() - 1; i++)
	//{
	//	derivative = w_const * (P[i + 1] - P[i - 1]) / (2 * params.h);
	//	integral_mst += params.h * t * derivative;
	//	integral_second += params.h * t * t * derivative;
	//	t += params.h;
	//}
	
	//if ((integral_second - pow(integral_mst, 2)) < 0)
	//{
	//	std::cout << "hi";
	//	*sd = -sqrt(std::abs(integral_second - pow(integral_mst, 2)));
	//}
	//else
	double mmst = pow(integral_mst, 2);
	double tempo = integral_second - mmst;
	std::cout << tempo << std::endl;
	*sd = sqrt(integral_second - mmst);
	*mst = integral_mst;
}

void CountMST()
{
	double i0 = 0.9;
	double h = 0.01;
	double A = 0.3;
	double gamma = 0.001;
	double* mst = new double;
	double* sd = new double;
	//double cc = 0.001 - 0.0002;

	//for (double k = 0.001; k >= 0.0001; k -= 0.0002)
	//{
	int exp_ind = 2;
	double k =gamma;
	std::string f("C:\\Users\\A\\Desktop\\graphs3\\mst" + std::to_string(exp_ind) + "_" + std::to_string(k) + ".txt");
	std::string f2("C:\\Users\\A\\Desktop\\graphs3\\sd" + std::to_string(exp_ind) + "_" + std::to_string(k) + ".txt");
	std::ofstream out(f);
	std::ofstream out2(f2);
	for (double i = 0.01; i <= 1.2; i += 0.01)
	{
		auto time1 = std::chrono::steady_clock::now();

		CalculatePropability(k, i, 200, h, A, i0, mst, sd);
		out << "t = " << i << "\tx = " << *mst << std::endl;
		out2 << "t = " << i << "\tx = " << *sd << std::endl;

		auto time2 = std::chrono::steady_clock::now();
		std::cout << "i = " << i << " done. Time: " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count() << " msec." << std::endl;
	}
	out.close();
	out2.close();
	//}

	delete mst;
	delete sd;
}

int main()
{
	CountMST();
	std::cout << "////////////////////" << std::endl;


	return 0;
}






//using namespace std;
//void histogram(std::vector<double> mass)
//{
//	int n = mass.size();
//	int max, min;
//	double width = 0;
//
//	double ccount = mass.size() / 100;
//
//	min = max = mass[0];
//	for (int i = 0; i < n; i++)
//	{
//		if (mass[i] >= max)
//		{
//			max = mass[i];
//		}
//		if (mass[i] <= min)
//		{
//			min = mass[i];
//		}
//	} //
//
//
//	vector <int> gist(ccount);
//	width = (max - min) / ccount;
//	//cout << width;
//	for (int i = 0; i < ccount; i++)
//	{
//		for (int j = 0; j < n; j++)
//		{
//			if ((mass[j] >= min + i * width) && (mass[j] <= min + (i + 1) * width))
//				gist[i]++;
//		}
//	}
//
//
//	std::ofstream out("C:\\Users\\A\\Desktop\\graphs2\\___n1.txt");
//	for (int i = 0; i < ccount; i++)
//	{
//		out << "t = " << min + i * width << "\tx = " << gist[i] << std::endl;
//		//cout << min + i * width << "| ";
//		//for (int j = 0; j < gist[i]; j++)
//		//{
//		//    //cout << "* ";
//		//}
//		//cout << endl;
//	}
//	out.close();
//}



//
//std::vector<double> GetStormerVerlet2nd(double _i0 = 0.5, double _T = 20, double _h = 0.01, double _omega = 0.62, unsigned rand_iter = 0)
//{
//	static long idum = -1;
//
//	double A = 0.05;
//	double omega = _omega;
//	double gamma = 0.001;
//	double alpha = 0.1;
//	double i0 = _i0;
//
//	double t = 0;
//	double t_max = _T;
//	double h = _h;
//
//	double fi_current = asin(i0);
//	double fi_next = 0;
//	double v_current = 0;
//	double v_next = 0;
//
//	double b = 1.0 / (1.0 + alpha * (h * 0.5));
//	double a = b / (1.0 + alpha * (h * 0.5));
//
//	std::vector<double> x;
//
//	double save_range = 1;
//	double tmp = 0;
//	//double noise_intensity = sqrt(2.0 * alpha * gamma * h);
//
//	while (t <= t_max - h)
//	{
//		double it = i0 + A * sin(omega * t);
//		double gas = gasdev(&idum);
//		double noise = sqrt(2.0 * alpha * gamma * h) * gas;
//
//		fi_next = fi_current + h * b * v_current - b * (h * h * 0.5) * (sin(fi_current) - it) + b * (h * 0.5) * noise;
//		v_next = a * v_current - (h * 0.5) * (a * sin(fi_current) + sin(fi_next)) + b * it * h + b * noise;
//
//		x.push_back(fi_current);
//
//		t += h;
//
//		fi_current = fi_next;
//		v_current = v_next;
//	}
//
//	return x;
//}