#include "../include/Regression.h"
#include "../include/Classification.h"
#include "../include/PMC.h"
#include <Eigen/Dense>

// Utilisation de Eigen::Map pour mapper les tableaux C vers des objets Eigen
// (permet d'éviter les copies de données)
using EigenMapMatrix = Eigen::Map<Eigen::MatrixXd>;
using EigenMapVector = Eigen::Map<Eigen::VectorXd>;



// classification




extern "C" __declspec(dllexport)
void trainClassification(double* X_data, double* Y_data,int N, int D,double learning_rate,int num_iteration,double* out_W_and_b)
{
    EigenMapMatrix X_map(X_data, N, D);
    EigenMapVector Y_map(Y_data, N);

    Classification model(D);

    // configurer les hyperparamètres avant l'entraînement
    model.set_learning_rate(learning_rate);
    model.set_iteration(num_iteration);

    // entraînement (Règle de Rosenblatt)
    model.updateWeights(X_map, Y_map); // Exécute la boucle d'entraînement

    // récupérer les résultats
    out_W_and_b[0] = model.get_bias();
    EigenMapVector W_out(out_W_and_b + 1, D);
    W_out = model.get_weight();
}

extern "C" __declspec(dllexport)
void predictClassification(double* X_test_data, int N_test, int D, double bias, double* W_data, double* output_prediction)
{
    EigenMapMatrix X_test(X_test_data, N_test, D);
    EigenMapVector W_map(W_data, D);

    Classification model(D);

    model.prediction(X_test);

    // recalculer la prédiction : y = Sign(X*W + b)
    Eigen::VectorXd scores = X_test * W_map;
    scores.array() += bias;
    Eigen::VectorXd predictions = scores.unaryExpr([](double x) {
        return x >= 0.0 ? 1.0 : -1.0;
        });

    EigenMapVector(output_prediction, N_test) = predictions;
}
//regression
extern "C" __declspec(dllexport)
void trainRegression(double* X_data, double* Y_data, int N, int D, double* out_W_and_b)
{
    EigenMapMatrix X_map(X_data, N, D);
    EigenMapVector Y_map(Y_data, N);

    Regression model(D);
    model.updateWeights(X_map, Y_map);

    // Récupérer les résultats
    out_W_and_b[0] = model.get_bias();
    EigenMapVector W_out(out_W_and_b + 1, D);
    W_out = model.get_weight();
}

extern "C" __declspec(dllexport)
void predictRegression(double* X_test_data, int N_test, int D, double bias, double* W_data, double* output_prediction)
{
    EigenMapMatrix X_test(X_test_data, N_test, D);
    EigenMapVector W_map(W_data, D);
    
    // recalculer la prédiction : y = X*W + b
    Eigen::VectorXd scores = X_test * W_map;
    scores.array() += bias;
    EigenMapVector(output_prediction, N_test) = scores;
}


// Perceptron Multi Couche
extern "C" __declspec(dllexport)
void* createPMC(int* npl_data, int npl_size) {

    // convertir le tableau C en vecteur C++
    std::vector<int> npl;

    for (int i = 0; i < npl_size; i++) 
    {
        npl.push_back(npl_data[i]);
    }

	// créer l'objet sur le heap car on utilise new
    PMC* model = new PMC(npl);

    
	return (void*)model; // cast en void* pour l'exporter
}


extern "C" __declspec(dllexport)
void trainPMC(void* modelPtr, double* X_flat, double* Y_flat, int nb_samples, int input_size, int output_size, int iteration, double learning_rate, bool is_classification)
{
    PMC* model = static_cast<PMC*>(modelPtr);
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> outputs;
    for (int i = 0; i < nb_samples; i++)
    {
        double* ptr_x = X_flat + (i * input_size);
        double* ptr_y = Y_flat + (i * output_size);

        inputs.push_back(Eigen::Map<Eigen::VectorXd>(ptr_x, input_size));
        outputs.push_back(Eigen::Map<Eigen::VectorXd>(ptr_y, output_size));

    }

    model->train(inputs, outputs, is_classification, iteration, learning_rate);
}

extern "C" __declspec(dllexport)
void predictPMC(void* modelPtr, double* input_data, int input_size, bool is_classification, double* output_data) 
{
    PMC* model = (PMC*)modelPtr;

    // créer le vecteur d'entrée
    Eigen::Map<Eigen::VectorXd> input_vec(input_data, input_size);

    // demander au modèle de prédire
    Eigen::VectorXd result = model->predict(input_vec, is_classification);

    // copier le résultat vers C#
    int out_size = result.size();
    for (int i = 0; i < out_size; i++) {
        output_data[i] = result(i);
    }
}

extern "C" __declspec(dllexport)
void deletePMC(void* model_ptr) 
{
    if (model_ptr != nullptr) //si c'est different de nullptr
    {
        PMC* model = static_cast<PMC*>(model_ptr);
        delete model;
    }
}