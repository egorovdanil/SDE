#pragma once
#include <iostream>
#include <vector>

#define LIFE_TIME_BREAK 1
#define PI 3.14159265

void GetNoise(float* r, int size, int seed);
void CalculatePropability(float gamma, float omega, float T, float h, float A, float i0, double* mst, double* sd);
void CountMST();
void histogram(std::vector<double> mass);
