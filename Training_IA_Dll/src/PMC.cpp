#include "../include/PMC.h"
#include <Eigen/Dense>


////// neurons_per_layer = { donnes d'entree , couche cachée, donnes de sortie }

PMC::PMC(const std::vector<int>& neurons_per_layer)
	: m_neurons_per_layer(neurons_per_layer),
	m_bias(0.0),                // initialisation de m_bias
	m_learning_rate(0.01)       // initialisation de m_learning_rate 
{
	int L = neurons_per_layer.size() - 1; // nombre de couches de poids

	for (int i = 0; i < L; i++)
	{
		int input_size = m_neurons_per_layer[i] + 1; // debut de la couche entree + le biais 
		int output_size = m_neurons_per_layer[i + 1]; //  la couche cachée ou fin 

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

		int is_last_layer = (i == L - 1); // si c'est la dernière couche

		//  Tanh SI : classification ou  Pas la dernière couche
		if (is_classification || !is_last_layer)
		{
			//capture le résultat retourné 
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

void PMC::train(const std::vector<Eigen::VectorXd>& all_sample_inputs, const std::vector<Eigen::VectorXd>& all_samples_expected_outputs, bool is_classification, int num_iterations, double learning_rate)
{

	for (int iter = 0; iter < num_iterations; iter++) {

		int k = std::rand() % all_sample_inputs.size();
		
		const Eigen::VectorXd& inputs_k = all_sample_inputs[k]; // Xij
		const Eigen::VectorXd& expected_outputs_k = all_samples_expected_outputs[k]; // Yij

		Eigen::VectorXd predicted_outputs = propagate(inputs_k, is_classification); // mise a jour de Xij

		// Calcul des deltas pour la couche de sortie (étape 1 du pdf)
		int L = m_weight.size();
		Eigen::VectorXd output_errors = predicted_outputs - expected_outputs_k;

		if (is_classification)
		{
			m_deltas[L].head(output_errors.size()) = (1 - predicted_outputs.array().square()) * output_errors.array(); // classification on prends la tete (head) car pas de biais
		}
		else
		{
			m_deltas[L].head(output_errors.size()) = output_errors; // régression  on prends la tete (head) car pas de biais
		}
		//étape de rétropropagation(étape 2 du pdf)
		for (int i=L-1; i >=0 ; i--)
		{

			Eigen::VectorXd next_layer = m_deltas[i + 1].head(m_neurons_per_layer[i + 1]); // sans biais
			Eigen::VectorXd total = m_weight[i].transpose() * next_layer;

			total = (1.0 - m_outputs[i].array().square()) * total.array();
			m_deltas[i].head(total.size()) = total;

		}

		// Mise à jour des poids (étape 3 du pdf)
		for (int i = 0; i <L; i++)
		{
			m_weight[i] -= learning_rate * (m_deltas[i + 1].head(m_neurons_per_layer[i + 1]) * m_outputs[i].transpose()); // dans delta on prends la tete (head) car pas de biais

		}
		
	}
	

}

