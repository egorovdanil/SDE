#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include "StormerVerlet.h"

#define LIFE_TIME_BREAK 1
#define PI 3.14159265

std::pair<double, double> CalculatePropability(Parameters params);
void MakeCurveByOmega();
