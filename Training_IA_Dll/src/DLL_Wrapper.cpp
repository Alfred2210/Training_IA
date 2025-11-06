#include "../include/Regression.h"
#include "../include/Classification.h"
#include <Eigen/Dense>

// Utilisation de Eigen::Map pour mapper les tableaux C vers des objets Eigen
// (permet d'éviter les copies de données)
using EigenMapMatrix = Eigen::Map<Eigen::MatrixXd>;
using EigenMapVector = Eigen::Map<Eigen::VectorXd>;



// classification


extern "C" __declspec(dllexport)
void PredictClassification(double* X_test_data, int N_test, int D, double bias, double* W_data, double* output_prediction)
{
    EigenMapMatrix X_test(X_test_data, N_test, D);
    EigenMapVector W_map(W_data, D);

    // Recalculer la prédiction : y = Sign(X*W + b)
    Eigen::VectorXd scores = X_test * W_map;
    scores.array() += bias;
    Eigen::VectorXd predictions = scores.unaryExpr([](double x) {
        return x >= 0.0 ? 1.0 : -1.0;
        });

    EigenMapVector(output_prediction, N_test) = predictions;
}

extern "C" __declspec(dllexport)
void TrainClassification(double* X_data, double* Y_data,int N, int D,double learning_rate,int num_iteration,double* out_W_and_b)
{
    EigenMapMatrix X_map(X_data, N, D);
    EigenMapVector Y_map(Y_data, N);

    Classification model(D);

    // Configurer les hyperparamètres avant l'entraînement
    model.set_learning_rate(learning_rate);
    model.set_iteration(num_iteration);

    // Entraînement (Règle de Rosenblatt)
    model.updateWeights(X_map, Y_map); // Exécute la boucle d'entraînement

    // Récupérer les résultats
    out_W_and_b[0] = model.get_bias();
    EigenMapVector W_out(out_W_and_b + 1, D);
    W_out = model.get_weight();
}

//regression

extern "C" __declspec(dllexport)
void PredictRegression(double* X_test_data, int N_test, int D, double bias, double* W_data, double* output_prediction)
{
    EigenMapMatrix X_test(X_test_data, N_test, D);
    EigenMapVector W_map(W_data, D);
    
    // recalculer la prédiction : y = X*W + b
    Eigen::VectorXd scores = X_test * W_map;
    scores.array() += bias;
    EigenMapVector(output_prediction, N_test) = scores;
}

extern "C" __declspec(dllexport)
void TrainRegression(double* X_data, double* Y_data, int N, int D, double* out_W_and_b)
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

