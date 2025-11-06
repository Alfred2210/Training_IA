#include "../include/LinearModel.h""


void LinearModel::set_learning_rate(double learing_rate)
{
	m_learning_rate = learing_rate;
}
void LinearModel::set_iteration(int iteration)
{
	m_iteration = iteration;
}
Eigen::VectorXd LinearModel::get_weight() const
{
	return m_weight;
}

double LinearModel::get_bias() const
{
	return m_bias;
}