#pragma once

#include "../framework.h"
#include <vector>
#include <Eigen/Dense>


class EXPORT_API LinearModel
{

public:

	LinearModel(int features)
		: m_weight(Eigen::VectorXd::Zero(features)), m_bias(0.0) {}

	virtual Eigen::VectorXd prediction(const Eigen::MatrixXd& X) const = 0;
	virtual void updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, double learning_rate)  = 0;

protected:
	Eigen::VectorXd m_weight;
	double m_bias;

};

