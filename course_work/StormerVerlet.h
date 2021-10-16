#pragma once
#include <iostream>
#include <vector>

struct StormerVerletParams
{
	float A;
	float omega;
	float gamma;
	float alpha;
	float i0;
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
	std::vector<float> it;

	StormerVerletParams();
	void Cinfigure();
	void CalculateCoefs();
	void CalculateItByTime();
};

void GetStormerVerlet(float* gas, std::vector<float>& P, StormerVerletParams& params, int id);