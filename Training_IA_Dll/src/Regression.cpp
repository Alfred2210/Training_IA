#include "../include/Regression.h"



Eigen::VectorXd Regression::prediction(const Eigen::MatrixXd& X) const
{

	Eigen::VectorXd scores = X * m_weight;
	scores.array() += m_bias;
	return scores;
}

void Regression::updateWeights(const Eigen::MatrixXd& X, const Eigen::VectorXd& Y)
{
	// N nombre d'exemples (lignes de X)
	int N = X.rows();
	// D  nombre de caractéristiques (colonnes de X)
	int D = X.cols();

	Eigen::MatrixXd Xt (N, D + 1);

	// on remplit la matrice Xt
	// première colonne = 1 (pour le biais)
	Xt.col(0).setOnes();
	// colonnes suivantes = Données originales
	Xt.rightCols(D) = X; 

	Eigen::MatrixXd X_carre = Xt.transpose() * Xt; 
	Eigen::VectorXd XT_Y = Xt.transpose() * Y;    

	//opearation inverse
	//Eigen::VectorXd W_full = X_carre.inverse()*XT_Y; -> fait crash 
	Eigen::VectorXd W_full = X_carre.ldlt().solve(XT_Y);
	m_bias = W_full[0];
	m_weight = W_full.tail(D);
}
