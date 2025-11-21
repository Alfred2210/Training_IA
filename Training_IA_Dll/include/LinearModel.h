#pragma once

#include "../framework.h"
#include <vector>
#include <Eigen/Dense>


class EXPORT_API LinearModel
{

public:

	LinearModel(int features)
		: m_weight(Eigen::VectorXd::Zero(features)), m_bias(0.0) {
	}
	virtual ~LinearModel() = default;
	virtual Eigen::VectorXd prediction(const Eigen::MatrixXd& X) const = 0;
	virtual void updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) = 0;

	void set_learning_rate(double learing_rate);
	void set_iteration(int iteration);
	Eigen::VectorXd get_weight() const;
	double get_bias() const;

protected:
	Eigen::VectorXd m_weight;
	double m_bias;
	double m_learning_rate;
	int m_iteration;

};

