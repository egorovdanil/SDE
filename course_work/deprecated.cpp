#include "common.h"

std::vector<double> GetStormerVerlet2nd(std::vector<double> gas, std::vector<double>& x_array, double _i0 = 0.7, double _T = 20, double _h = 0.01, double _omega = 0.62)
{
	static long idum = -1;

	double A = 0.05;
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

	double b = 1.0 / (1.0 + alpha * (h * 0.5));
	double a = b / (1.0 + alpha * (h * 0.5));

	std::vector<double> x;

	double save_range = 1;
	double tmp = 0;
	//double noise_intensity = sqrt(2.0 * alpha * gamma * h);

	size_t size = x_array.size();
	for (size_t i = 1; i < size; i++)
	{
		double it = i0 + A * sin(omega * t);
		double noise = sqrt(2.0 * alpha * gamma * h) * gas[i];

		fi_next = fi_current + h * b * v_current - b * (h * h * 0.5) * (sin(fi_current) - it) + b * (h * 0.5) * noise;
		v_next = a * v_current - (h * 0.5) * (a * sin(fi_current) + sin(fi_next)) + b * it * h + b * noise;

		x_array[i] = fi_current;
		//t += h;

		fi_current = fi_next;
		v_current = v_next;
	}

	return x;
}

void GetHeun2nd(std::vector<double> gas, std::vector<double>& x_array, double _i0 = 0.5, double _T = 20, double _h = 0.01, double _omega = 0.62, int i = 0)
{
	double A = 0.6;
	double omega = _omega;
	double gamma = 0.001;
	double alpha = 0.1;
	double i0 = _i0;

	double t = 0;
	double t_max = _T;
	double h = _h;
	int first_step = 1;

	double x1;
	double v1;
	double x0;
	double v0;
	double x = asin(i0);
	double v = 0;
	x_array[0] = x;

	int index_exp = i;

	size_t size = x_array.size();
	for (size_t i = 1; i < size; i++)
	{
		double it = i0 + A * sin(omega * t);
		double noise = sqrt(2.0 * alpha * gamma * h) * gas[i];

		x0 = x;
		v0 = v;

		x1 = x0 + h * v0;
		v1 = v0 - h * (alpha * v0 + sin(x0) - it) + noise;

		x = x0 + h * 0.5 * (v1 + v0);
		v = v0 - h * 0.5 * (alpha * (v1 + v0) + sin(x0) - 2 * it + sin(x1)) + noise;

		if (first_step)
		{
			t += h * 0.5;
			first_step = 0;
		}
		else
		{
			t += h;
		}

		x_array[i] = x;

	}

	index_exp++;
}