#include <cassert>

#include "Neuron.h"
#include "CostFunc.h"

using namespace std;
using namespace ML;

///////////////////////////////////////////////////////////////////////////////
// Mean-squared error implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * Calculate errors for the output layer of the NN
 */
void MSE::calcOutputError(const vector<double>& labels,
													const vector<double>& activations,
													const vector<double>& sums,
													vector<double>& errors)
{
	assert(labels.size() == activations.size());
	assert(labels.size() == sums.size());
	assert(labels.size() == errors.size());

	for(size_t i = 0; i < labels.size(); i++)
		errors[i] = (activations[i] - labels[i]) * _neuron->gradient(sums[i]);
}

/**
 * Calculate errors for a hidden layer of the NN
 */
void MSE::calcHiddenError(const matrix<double>& weights,
													const vector<double>& nextErrors,
													const vector<double>& sums,
													vector<double>& errors)
{
	assert(weights.cols() == nextErrors.size());
	assert(weights.rows() == sums.size());
	assert(errors.size() == sums.size());

	for(size_t i = 0; i < weights.rows(); i++)
	{
		for(size_t j = 0; j < weights.cols(); j++)
			errors[i] += weights(i, j) * nextErrors[j];
		errors[i] *= _neuron->gradient(sums[i]);
	}
}

/**
 * Calculate gradients for all weights in a layer
 */
void MSE::calcWeightGradients(const vector<double>& errors,
															const vector<double>& prevActivations,
															matrix<double>& gradients)
{
	assert(gradients.rows() == prevActivations.size());
	assert(gradients.cols() == errors.size());

	for(size_t i = 0; i < gradients.rows(); i++)
		for(size_t j = 0; j < gradients.cols(); j++)
			gradients(i, j) = prevActivations[i] * errors[j];
}

