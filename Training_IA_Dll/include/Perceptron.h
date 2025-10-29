#pragma once
#include "../include/LinearModel.h"
#include <Eigen/Dense>

class Perceptron : public LinearModel
{

public :
	Perceptron(int features)
		: LinearModel(features) {
	}
	Eigen::VectorXd prediction(const Eigen::MatrixXd& X) const override;
	void updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, double learning_rate) override;

};

