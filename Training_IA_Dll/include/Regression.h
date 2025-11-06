#pragma once

#include "../include/LinearModel.h"
#include <Eigen/Dense>

class EXPORT_API Regression : public LinearModel
{

	public :
	Regression(int features)
		: LinearModel(features) {
	}
	Eigen::VectorXd prediction(const Eigen::MatrixXd& X) const override;
	void updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y) override;
};

