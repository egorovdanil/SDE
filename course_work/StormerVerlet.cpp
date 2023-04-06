#include "common.h"
#include "StormerVerlet.h"

double StormerVerlet(Parameters params) {
	//std::string id = std::to_string(std::time(0));
	//std::string file = "C:\\Users\\A\\source\\SDE\\graphs\\signal_" + id + "_" + std::to_string(params.signal.size()) + ".txt";
	//std::ofstream out(file);

	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<> normal{ 0.0, 1.0 };

	double result = params.t_max;
	float fi_current = asin(params.bias_current);
	float sin_fi = 0;
	float fi_next = 0;
	float v_current = 0;
	float v_next = 0;
	float noise = 0;
	double t = 0;
	float it = 0;
	float signal = 0;

	auto barrier = [&] {
		if (!(fi_current <= params.h_barrier && fi_current >= params.l_barrier))
		{
			result = t;
			return true;
		}
		return false;
	};

	auto step = [&] {
		it = params.bias_current + signal;

		noise = params.noise_intensity * normal(gen);
		sin_fi = sin(fi_current);

		fi_next = fi_current + params.c_bh * v_current - params.c_bh_squar * (sin_fi - it) + params.c_bh_half * noise;
		v_next = params.a * v_current - params.c_h_half * (params.a * sin_fi + sin(fi_next)) + it * params.c_bh + params.b * noise;

		fi_current = fi_next;
		v_current = v_next;

		t += params.h;
	};

	for (size_t i = 0; i < params.size; i++) {
		if (barrier()) break;

		signal = params.Signal();
		//out << t << "\t" << signal << std::endl;

		step();
	}

	return result;
}

