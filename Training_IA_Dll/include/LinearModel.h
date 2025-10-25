#pragma once

#include "../framework.h"
#include <vector>
#include "../include/Matrix.h"


class EXPORT_API LinearModel
{

public:

	LinearModel(int features)
		: m_weight(features,0.0), m_bias(0.0) {}

	Matrix y_prediction(const Matrix& X);
	void updateWeights(const Matrix& X, const Matrix& Y, double learning_rate);
	Matrix normalize(const Matrix& X);
private:
	std::vector<double> m_weight;
	double m_bias;

};

