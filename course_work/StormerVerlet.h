#pragma once
#include <iostream>
#include <vector>
#include <map>
#define PI 3.14159265

inline std::map<int, int> poisson_statistic;

namespace Signal {

    enum class StatisticType {
        Poisson = 0,
        SuperPoisson,
        SubPoisson,
    };

    enum class Type {
        NoSignal = 0,
        SineContinuos,
        SineContinuos2,
        HalfSineImpulse,
        RectImpulse,
        TrapezImpulse,
    };

    class Base {
    protected:
        size_t time;
        size_t signal_duration;
    public:
        double amplitude;
        double omega;
        double t_step;
        size_t shift;

        Base() {
            amplitude = 0.5;
            omega = 0.5;
            t_step = 0.01;
            signal_duration = UINT_MAX;
            time = 0;
            shift = 0;
        }

        virtual void Configure() {
            signal_duration = (1 / omega) / (t_step);
        }

        virtual void Reset() {
            time = 0;
        }

        virtual double operator()() {
            return 0;
        }
    };

    class SineContinuos : public Base {
        double signal_step;

    public:
        void Configure() override {
            signal_duration = UINT_MAX;
            signal_step = PI / ((1 / omega) / (t_step));
        }

        double operator()() override {
            double signal_time = time++ * t_step;
            return amplitude * std::sin(omega * signal_time);
        }
    };

    class SineContinuos2 : public Base {
        double signal_step;

    public:
        void Configure() override {
            signal_duration = UINT_MAX;
            signal_step = PI / ((1 / omega) / (t_step));
        }

        double operator()() override {
            return amplitude * std::sin(signal_step * time++);
        }
    };

    class HalfSineImpulse : public Base {
        double signal_step;

    public:
        void Configure() override {
            Base::Configure();
            signal_step = PI / signal_duration;
            time = 0;
        }

        double operator()() override {
            if (shift > 0) {
                shift--;
                return 0;
            }

            if (time > signal_duration)
                return 0;
            return amplitude * std::sin(signal_step * time++);
        }
    };

    class RectImpulse : public Base {
    public:
        void Configure() override {
            Base::Configure();
        }

        double operator()() override {
            if (shift > 0) {
                shift--;
                return 0;
            }

            if (time > signal_duration)
                return 0;
            time++;
            return amplitude;
        }
    };

    class TrapezImpulse : public Base {
        double duration;
        double t_grow;
        double t_middle;
        double t_decrease;
    public:
        double grow;
        double decrease;
        double amplitude2;

        TrapezImpulse() {
        }

        void Configure() override {
            Base::Configure();
            time = 0;

            grow = 0.2;
            decrease = 0.2;
            amplitude2 = amplitude * 1;

            duration = (1 / omega);
            t_grow = duration * grow;
            t_middle = duration - duration * decrease;
            t_decrease = duration;

        }

        double operator()() override {
            if (shift > 0) {
                shift--;
                return 0;
            }

            double t = time++ * t_step;
            if (t >= t_decrease) {
                return 0;
            }
            else if (t < t_grow) {
                return amplitude * t / t_grow;
            }
            else if (t < t_middle) {
                return  (t - t_grow) * (amplitude2 - amplitude) / (t_middle - t_grow) + amplitude;
            }
            else if (t < t_decrease) {
                return amplitude2 * (t - t_decrease) / (t_middle - t_decrease);
            }
            return 0;
        }
    };
}


struct RealParameters
{
	double T = 100;
	double Ic = 0.0851;
	double Ib = 0.0415;
	double Cp = 0.036;
	double Rn = 3400;
	double A = 0;
	double Fq = 0;
	double Wp = 0;

	double seconds = 0.001;
};

class Parameters
{
public:
	float amplitude;
	float omega;
	float gamma;
	float alpha;
	float bias_current;
	float t;
	float t_max;
	float h;

	float b;
	float a;

	float noise_intensity;
	float c_h_half;
	float c_bh_half;
	float c_bh;
	float c_bh_squar;

	float h_barrier;
	float l_barrier;

	size_t size;

	RealParameters real;

	std::vector<std::unique_ptr<Signal::Base>> signal;
	Signal::Type signal_type;
    Signal::StatisticType stat_type;
    bool single_pulse;
    size_t signal_shift;
    size_t phase_acceleration;
    int stat_mean;

	Parameters() {
		amplitude = 0.6;
		omega = 0.5;
		gamma = 0.001;
		alpha = 0.1;
		bias_current = 0.5;
		t = 0;
		t_max = 20;
		h = 0.01;
        signal_type = Signal::Type::NoSignal;
        stat_type = Signal::StatisticType::Poisson;
        single_pulse = false;
        signal_shift = 0;
        phase_acceleration = 20;
        stat_mean = 1;

		Configure();
	}

    Parameters(const Parameters& params) {
        amplitude = params.amplitude;
        omega = params.omega;
        gamma = params.gamma;
        alpha = params.alpha;
        bias_current = params.bias_current;
        t = params.t;
        t_max = params.t_max;
        h = params.h;
        signal_type = params.signal_type;
        stat_type = params.stat_type;
        single_pulse = params.single_pulse;
        signal_shift = params.signal_shift;
        phase_acceleration = params.phase_acceleration;
        stat_mean = params.stat_mean;

        Configure();
    }

	void Configure() {
        if (amplitude - bias_current > 1 || amplitude - bias_current < -1) {
            std::cout << "wrong asin arg\n";
        }
		h_barrier = PI + asin(amplitude - bias_current);
		// h_barrier = PI; // temporary
		l_barrier = -h_barrier;

		b = 1.0 / (1.0 + alpha * (h * 0.5));
		a = b / (1.0 + alpha * (h * 0.5));

		noise_intensity = sqrt(2.0 * alpha * gamma * h);
		c_h_half = h * 0.5;
		c_bh_half = b * c_h_half;
		c_bh = h * b;
		c_bh_squar = c_bh * c_h_half;
		if (c_bh_half != (b * (h * 0.5f))) std::cout << c_bh_half << " " << b * (h * 0.5) << "Err\n" << std::endl;

		size = t_max / h;

        std::random_device rd{};
        std::mt19937 gen{ rd() };
        std::poisson_distribution<> poisson(stat_mean);
        std::negative_binomial_distribution<> super_poisson(stat_mean + 1, 0.5);
        std::binomial_distribution<> sub_poisson(stat_mean * 2 - 1, 0.5);
        std::uniform_int_distribution<> uniform(phase_acceleration, size);

        if (single_pulse) {
            signal_shift = uniform(gen);
        }

        //std::cout << "signal_shift: " << signal_shift << "\n";

        size_t pulses_num = 0;

        switch (stat_type)
        {
        case Signal::StatisticType::Poisson:
            pulses_num = poisson(gen);
            break;
        case Signal::StatisticType::SuperPoisson:
            pulses_num = super_poisson(gen);
            break;
        case Signal::StatisticType::SubPoisson:
            pulses_num = sub_poisson(gen);
            break;
        default:
            break;
        }

        ++poisson_statistic[pulses_num];

		switch (signal_type)
		{
		case Signal::Type::SineContinuos:
            signal.push_back(std::make_unique<Signal::SineContinuos>());
			break;
        case Signal::Type::SineContinuos2:
            signal.push_back(std::make_unique<Signal::SineContinuos2>());
            break;
		case Signal::Type::HalfSineImpulse:
            for (size_t i = 0; i < pulses_num; i++) {
                signal.push_back(std::make_unique<Signal::HalfSineImpulse>());
            }
			break;
		case Signal::Type::RectImpulse:
            for (size_t i = 0; i < pulses_num; i++) {
                signal.push_back(std::make_unique<Signal::RectImpulse>());
            }
			break;
        case Signal::Type::TrapezImpulse:
            for (size_t i = 0; i < pulses_num; i++) {
                signal.push_back(std::make_unique<Signal::TrapezImpulse>());
            }
            break;
		case Signal::Type::NoSignal:
		default:
			signal.push_back(std::make_unique<Signal::Base>());
			break;
		}

        for (size_t i = 0; i < signal.size(); i++) {
            signal[i]->amplitude = amplitude;
            signal[i]->omega = omega;
            signal[i]->t_step = h;
            if (!single_pulse) {
                signal_shift = uniform(gen);
            }
            signal[i]->shift = signal_shift;
            signal[i]->Configure();
            //signal[i]->shift = 200 * i + 20 * i;
        }
	}

    double Signal() {
        double res = 0;
        for (size_t i = 0; i < signal.size(); i++) {
            res += signal[i]->operator()();
        }
        return res;
    }

	void ConfigureFromReal() {
		const long double c_s2eh = 55122907194.91327; // sqrt(2 * e / h)* sqrt(CI / CC)
		const long double c_1s2eh = 18.141278297678028; // c_s2eh / (2 * e * CI / h)
		const long double c_2ekh = 4.195150167862804e-05; // 2 * e * k / h

		real.Wp = c_s2eh * sqrt(real.Ic / real.Cp) / (2.0 * PI);

		bias_current = real.Ib / real.Ic;
		amplitude = real.A * real.Ic;
		omega = real.Fq / real.Wp;
		alpha = c_1s2eh / (real.Rn * sqrt(real.Ic * real.Cp));
		gamma = c_2ekh * real.T / real.Ic;

		t_max = real.Wp * real.seconds;

		Configure();
	}
};

double StormerVerlet(Parameters params);