#include "../include/Perceptron.h"



Eigen::VectorXd Perceptron::prediction(const Eigen::MatrixXd& X) const
{
	// Y = X*W + b
	Eigen::VectorXd scores = X * m_weight;
	scores.array() += m_bias;

	// si score >= 0 alors 1 sinon -1
	Eigen::VectorXd Y_prime = scores.unaryExpr([](double x) {
		return x >= 0.0 ? 1.0 : -1.0;
		});

	return Y_prime;

}

void Perceptron::updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, double learning_rate)
{
	//regle de rosenblatt
	for (int i = 0; i < X.rows(); ++i)
	{
		// Y = X*W + b
		double y_pred = X.row(i).dot(m_weight) + m_bias;

		// si score >= 0 alors 1 sinon -1
		double y_prediction = (y_pred >= 0.0) ? 1.0 : -1.0;

		// calcul de l'erreur
		double error = Y(i) - y_prediction;

		// mise à jour des poids et du biais si erreur
		if (error != 0.0)
		{
			m_weight += learning_rate * error * X.row(i).transpose();
			m_bias += learning_rate * error;
		}
	}
}