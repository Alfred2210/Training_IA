#pragma once
#include <Eigen/Dense>

class PMC
{
public:
	PMC(const std::vector<int>& neurons_per_layer);
	Eigen::VectorXd propagate(const Eigen::VectorXd& inputs, bool is_classification = true);
	Eigen::VectorXd predict(const Eigen::VectorXd& inputs, bool is_classification = true);
	void train(const std::vector<Eigen::VectorXd>& all_sample_inputs, const  std::vector <Eigen::VectorXd>& all_samples_expected_outputs, bool is_classification = true, int num_iterations =1000, double learning_rate = 0.01);
private :
	std::vector<int> m_neurons_per_layer;
	std::vector<Eigen::MatrixXd> m_weight;
	std::vector<Eigen::VectorXd> m_outputs;
	std::vector<Eigen::VectorXd> m_deltas;
	double m_bias;
	double m_learning_rate;
};

