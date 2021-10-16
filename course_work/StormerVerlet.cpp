#include "common.h"
#include "StormerVerlet.h"

void GetStormerVerlet(float* gas, std::vector<float>& P, StormerVerletParams& params, int id)
{
	float fi_current = asin(params.i0);
	float sin_fi = 0;
	float fi_next = 0;
	float v_current = 0;
	float v_next = 0;
	float noise = 0;

#ifdef DUMP_POINTS
	std::string f("C:\\Users\\A\\Desktop\\graphs6\\___57_" + std::to_string(id) + "_" + ".txt");
	std::ofstream out(f);
	float t = 0;
#endif

	for (size_t i = 0; i < params.size; ++i)
	{
		if (fi_current <= params.h_barrier && fi_current >= params.l_barrier)
		{
			P[i] += 1;
		}

#ifdef DUMP_POINTS
		out << "t = " << t << "\tx = " << fi_current << std::endl;
#endif

		noise = params.noise_intensity * gas[i];
		sin_fi = sin(fi_current);

		fi_next = fi_current + params.c_bh * v_current - params.c_bh_squar * (sin_fi - params.it[i]) + params.c_bh_half * noise;
		v_next = params.a * v_current - params.c_h_half * (params.a * sin_fi + sin(fi_next)) + params.it[i] * params.c_bh + params.b * noise;

		fi_current = fi_next;
		v_current = v_next;

#ifdef DUMP_POINTS
		t += params.h;
#endif
	}

#ifdef DUMP_POINTS
	out.close();
#endif
}

StormerVerletParams::StormerVerletParams()
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

void StormerVerletParams::Cinfigure()
{
	h_barrier = PI + asin(A - i0);
	l_barrier = -h_barrier;
	CalculateCoefs();
	CalculateItByTime();
}

void StormerVerletParams::CalculateCoefs()
{
	b = 1.0 / (1.0 + alpha * (h * 0.5));
	a = b / (1.0 + alpha * (h * 0.5));

	noise_intensity = sqrt(2.0 * alpha * gamma * h);
	c_h_half = h * 0.5;
	c_bh_half = b * c_h_half;
	c_bh = h * b;
	c_bh_squar = c_bh * c_h_half;
	if (c_bh_half != (b * (h * 0.5f))) std::cout << c_bh_half << " " << b * (h * 0.5) << "Err\n" << std::endl;

	size = t_max / h;
	it.resize(size);
};

void StormerVerletParams::CalculateItByTime()
{
	float t = 0;
	short first_step = 1;
	for (size_t i = 0; i < size; ++i)
	{
		it[i] = i0 + A * sin(omega * t);

		if (first_step)
		{
			t += h * 0.5;
			first_step = 0;
		}
		else
		{
			t += h;
		}
	}
}
