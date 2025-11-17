#pragma once
#include <Eigen/Dense>

class PMC
{
public:
	

	
	PMC(const std::vector<int>& neurons_per_layer);
	Eigen::VectorXd propagate(const Eigen::VectorXd& inputs, bool is_classification = true);
	Eigen::VectorXd predict(const Eigen::VectorXd& inputs, bool is_classification = true);
private :
	std::vector<int> m_neurons_per_layer;
	std::vector<Eigen::MatrixXd> m_weight;
	std::vector<Eigen::VectorXd> m_outputs;
	std::vector<Eigen::VectorXd> m_deltas;
	double m_bias;
	double m_learning_rate;
};

