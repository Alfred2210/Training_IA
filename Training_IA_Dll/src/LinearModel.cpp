#include "../pch.h"
#include "../include/LinearModel.h"
#include <iostream>

Matrix LinearModel::y_prediction(const Matrix& X)
{
	Matrix Y_prime{ X.getRows(), 1 };

	for (size_t i = 0; i < X.getRows(); i++)
	{

		double y_prediction = 0.0;
		for (size_t j = 0; j < X.getColums(); j++)
		{
			y_prediction += X(i, j) * m_weight[j];
		}
		y_prediction += m_bias;

		if (y_prediction >= 0.0)
			Y_prime(i, 0) = 1.0;
		else
			Y_prime(i, 0) = -1.0;

	}
	return Y_prime;
}


Matrix LinearModel::normalize(const Matrix& X)
{
	Matrix X_norm{ X.getRows(), X.getColums() };
	for (size_t j = 0; j < X.getColums(); j++)
	{
		double min = X(0, j);
		double max = X(0, j);
		for (size_t i = 0; i < X.getRows(); i++)
		{
			if (X(i, j) < min)
				min = X(i, j);
			if (X(i, j) > max)
				max = X(i, j);
		}
		double range = max - min;
		for (size_t i = 0; i < X.getRows(); i++)
		{
			if (range != 0)
				X_norm(i, j) = (X(i, j) - min) / range;
			else
				X_norm(i, j) = 0.0;
		}
	}
	return X_norm;
}

void LinearModel::updateWeights(const Matrix& X, const Matrix& Y, double learning_rate)
{

	for (size_t i = 0; i < X.getRows(); i++)
	{
		double y_pred = m_bias;
		for (size_t j = 0; j < X.getColums(); j++)
		{
			y_pred += X(i, j) * m_weight[j];
		}
		double y_prediction;
		if (y_pred >= 0.0)
		{
			y_prediction = 1.0;
		}
		else
		{
			y_prediction = -1.0;
		}

		if (Y(i, 0) != y_prediction)
		{
			for (size_t j = 0; j < X.getColums(); j++)
			{
				m_weight[j] += learning_rate * X(i, j) * Y(i, 0);	
			}
			m_bias += learning_rate * Y(i, 0);
		}

	}
}
