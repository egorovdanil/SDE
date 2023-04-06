#include <chrono>
#include <omp.h>

#include "common.h"
#include "StormerVerlet.h"

std::pair<double, double> CalculatePropability(Parameters params)
{
	const unsigned short numtasks = 16;

	size_t switch_num = 16000;

	if (switch_num % numtasks) {
		switch_num = switch_num / numtasks;
		//std::cout << "switch num changed to " << switch_num << std::endl;
	}

	std::vector<double> timings(switch_num);
	size_t part_count = switch_num / numtasks;

	omp_set_num_threads(numtasks);
#pragma omp parallel shared(numtasks, params)
	{
		short taskid = omp_get_thread_num();
		size_t shift = taskid * part_count;

		for (size_t i = 0; i < part_count; i++) {
			timings[shift + i] = StormerVerlet(params);
		}
	}

	double mst = 0;
	double sd = 0;
	for (auto time: timings) {
		mst += time;
	}
	mst = mst / switch_num;

	for (auto time: timings) {
		sd += (time - mst) * (time - mst);
	}
	sd = sqrt(sd / (switch_num - 1));
	return std::pair<double, double>(mst, sd);
}

void MakeCurveByOmega()
{
	std::string id = std::to_string(std::time(0));
	std::string file_name_params = "C:\\Users\\A\\source\\SDE\\graphs\\parameters_" + id + ".txt";
	std::string file_name_mst = "C:\\Users\\A\\source\\SDE\\graphs\\mst_" + id + ".txt";
	std::string file_name_sd = "C:\\Users\\A\\source\\SDE\\graphs\\sd_" + id + ".txt";
	std::string file_name_mst2sd = "C:\\Users\\A\\source\\SDE\\graphs\\mst2sd_" + id + ".txt";

	std::string file_name_poisson_stat = "C:\\Users\\A\\source\\SDE\\graphs\\ps_" + id + ".txt";
	//std::string file_name_poisson_stat_mean = "C:\\Users\\A\\source\\SDE\\graphs\\ps_mean_" + id + ".txt";

	std::ofstream out_params(file_name_params);
	std::ofstream out_mst(file_name_mst);
	std::ofstream out_sd(file_name_sd);
	std::ofstream out_mst2sd(file_name_mst2sd);

	std::ofstream out_ps(file_name_poisson_stat);
	//std::ofstream out_ps_mean(file_name_poisson_stat_mean);

	//params.amplitude = 0.01;
	//params.gamma = 0.001;
	//params.alpha = 0.1;
	//params.bias_current = 1;
	//params.t_max = 2000;
	//params.h = 0.01;
	//params.signal_type = Signal::Type::RectImpulse;

	//params.real.T = 10;
	//params.real.Ic = 0.4;
	//params.real.Ib = 0.2;
	//params.real.Cp = 7;
	//params.real.Rn = 108;
	//params.real.A = 1.5;
	//params.real.Fq = 2520000000;
	//params.real.seconds = 0.000001;
	//params.ConfigureFromReal();

	Parameters params;
	params.amplitude = 0.05;
	params.gamma = 0.001;
	params.alpha = 0.1;
	params.bias_current = 0.9;
	params.t_max = 1000;
	params.h = 0.01;
	params.signal_type = Signal::Type::RectImpulse;
	params.stat_type = Signal::StatisticType::SuperPoisson;
	params.stat_mean = 4;
	params.single_pulse = true;


	std::map<Signal::Type, std::string> Type2String = {
		{Signal::Type::NoSignal, "NoSignal"},
		{Signal::Type::SineContinuos, "SineContinuos"},
		{Signal::Type::SineContinuos2, "SineContinuos2"},
		{Signal::Type::HalfSineImpulse, "HalfSineImpulse"},
		{Signal::Type::RectImpulse, "RectImpulse"},
		{Signal::Type::TrapezImpulse, "TrapezImpulse"},
	};

	std::map<Signal::StatisticType, std::string> Stat2String = {
		{Signal::StatisticType::Poisson, "Poisson"},
		{Signal::StatisticType::SuperPoisson, "SuperPoisson"},
		{Signal::StatisticType::SubPoisson, "SubPoisson"},
	};

	out_params << "amplitude: " << params.amplitude << std::endl;
	out_params << "gamma: " << params.gamma << std::endl;
	out_params << "alpha: " << params.alpha << std::endl;
	out_params << "bias_current: " << params.bias_current << std::endl;
	out_params << "time: " << params.t_max << std::endl;
	out_params << "step: " << params.h << std::endl;
	out_params << "signal: " << Type2String[params.signal_type] << std::endl;
	out_params << "statistic: " << Stat2String[params.stat_type] << std::endl;
	out_params << "statistic mean: " << params.stat_mean << std::endl;

	out_mst << "omega" << "\t" << "mst" << std::endl;
	out_sd << "omega" << "\t" << "sd" << std::endl;

	size_t points = 0;
	auto begin = std::chrono::high_resolution_clock::now();
	//for (double Fq = 200000000; Fq < 2550000000; Fq += 21250000)
	for (double omega = 0.01; omega < 1.2; omega += 0.01)
	{
		//params.real.Fq = Fq;
		//params.ConfigureFromReal();
		params.omega = omega;
		params.Configure();
		auto res = CalculatePropability(params);
		out_mst << params.omega << "\t" << res.first << std::endl;
		out_sd << params.omega << "\t" << res.second << std::endl;
		out_mst2sd << params.omega << "\t" << res.first / res.second << std::endl;
		//out_mst << params.omega << "\t" << res.first / params.real.Wp << std::endl;
		//out_sd << params.omega << "\t" << res.second / params.real.Wp << std::endl;

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
		std::cout << "\n"; // "\r"; //
		std::cout << "omega: " << params.omega << "\ttime: " << elapsed.count() << " seconds";

		std::cout << "\nmst / sd: " << res.first / res.second << std::endl;
		++points;
	}
	std::cout << std::endl;

	std::cout << "points " << points << std::endl;

	for (auto x : poisson_statistic) {
		out_ps << x.first << '\t' << x.second / points << std::endl;
		//out_ps_mean << x.first << '\t' << x.second / poisson_statistic.size() << '\n';
	}

	//std::map<int, int> ps;
	//std::map<int, int> ps_mean;


	//for (auto omega : poisson_stat) {
	//	for (auto photons_num : omega.second) {
	//		ps[photons_num.first] += photons_num.second;
	//	}
	//}

	//for (auto x : ps) {
	//	out_ps << x.first << '\t' << x.second << '\n';
	//	out_ps_mean << x.first << '\t' << x.second / poisson_stat.size() << '\n';
	//}
}
