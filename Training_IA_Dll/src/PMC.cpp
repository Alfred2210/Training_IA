#include "../include/PMC.h"
#include <Eigen/Dense>


// neurons_per_layer = { donnes d'entree , couche chachee, donnes de sortie }

PMC::PMC(const std::vector<int>& neurons_per_layer)
	: m_neurons_per_layer(neurons_per_layer),
	m_bias(0.0),                // initialisation de m_bias
	m_learning_rate(0.01)       // initialisation de m_learning_rate 
{
	int L = neurons_per_layer.size() - 1;

	for (int i = 0; i < L; i++)
	{
		int input_size = m_neurons_per_layer[i] + 1; // debut de la couche entree + le biais 
		int output_size = m_neurons_per_layer[i + 1]; //  la couche cachee ou fin 

		// initialisation des poids avec des valeurs aléatoires
		Eigen::MatrixXd weight_matrix = Eigen::MatrixXd::Random(output_size, input_size); // destination et source == lignes = destination/output et colonnes = sources/input
		m_weight.push_back(weight_matrix);
	}

	// Initialisation des outputs et des deltas pour chaque couche
	for (size_t i = 0; i < m_neurons_per_layer.size(); i++)
	{

		int size_biais = m_neurons_per_layer[i] + 1; // +1 pour le biais

		m_deltas.push_back(Eigen::VectorXd::Zero(size_biais));
		m_outputs.push_back(Eigen::VectorXd::Zero(size_biais)); //X en python enfin outputs ici

	}

}

Eigen::VectorXd PMC::propagate(const Eigen::VectorXd& inputs, bool is_classification)
{

	m_outputs[0].head(inputs.size()) = inputs; // ajout des inputs dans la premiere couche
	m_outputs[0](inputs.size()) = 1.0;

	int L = m_weight.size();

	for (int i = 0; i < L; i++)
	{

		Eigen::VectorXd signal = m_weight[i] * m_outputs[i];
		Eigen::VectorXd x_result;

		int is_last_layer = (i == L - 1); // si c'est la derniere couche

		//  Tanh SI : classification ou  Pas la dernière couche
		if (is_classification || !is_last_layer)
		{
			//capture le résultat retourné !
			x_result = signal.unaryExpr([](double val) { return std::tanh(val); });
		}
		else
		{
			// régression sur la dernière couche : on garde le signal brut
			x_result = signal;
		}
		m_outputs[i + 1].head(x_result.size()) = x_result;

		// ajout du biais si ce n'est pas la dernière couche
		if (!is_last_layer) {
			m_outputs[i + 1](x_result.size()) = 1.0;
		}
	}
	// dernière couche : pas de biais ajouté
	Eigen::VectorXd final = m_outputs.back();
	return final.head(final.size() - 1);
}

Eigen::VectorXd PMC::predict(const Eigen::VectorXd& inputs, bool is_classification)
{
	propagate(inputs, is_classification);

	return m_outputs.back().head(m_neurons_per_layer.back()); // retourne les sorties sans le biais
}

