/**
 * Cost functions used for backpropagation
 *
 * Author: Rob Lyerly
 * Date: 9/1/2014
 */

#ifndef _COST_FUNC_H
#define _COST_FUNC_H

#include "matrix.h"
#include "vector.h"

namespace ML
{

/**
 * Cost function template which cost-function sub-classes should implement
 *
 * Provides the functionality needed to calculate partial derivatives and
 * backpropagate errors.
 */
class CostFunc
{
public:
	/**
	 * Implemented cost functions
	 */
	typedef enum CostFuncType {
		MSE
	} CostFuncType;

	CostFunc(Neuron* neuron)
		: _neuron(neuron) {}
	virtual ~CostFunc() {};

	/**
	 * Calculate error for the output layer L
	 *
	 * @param labels a vector of expected outputs from the neural network
	 * @param outputs a vector of actual outputs from the neural network
	 * @param sums a vector of weighted sums for L
	 * @return a vector in which to store the errors for L
	 */
	virtual void calcOutputError(const std::vector<double>& labels,
															 const std::vector<double>& nnOutputs,
															 const std::vector<double>& sums,
															 std::vector<double>& errors) = 0;

	/**
	 * Calculate error for a hidden layer l.  Note: these errors are equal to the
	 * gradients for the biases.
	 *
	 * @param weights weight matrix for l x l+1 layers
	 * @param nextErrors error vector for next layer, l+1
	 * @param sums a vector of weight sums for l
	 * @return a vector in which to store errors for l
	 */
	virtual void calcHiddenError(const std::matrix<double>& weights,
															 const std::vector<double>& nextErrors,
															 const std::vector<double>& sums,
															 std::vector<double>& errors) = 0;

	/**
	 * Calculate weight gradients for a given layer l
	 *
	 * @param errors previously calculated errors for layer l
	 * @param prevActivations activations for layer l-1
	 * @return a matrix which contains the weight gradients
	 */
	virtual void calcWeightGradients(const std::vector<double>& errors,
																	 const std::vector<double>& prevActivations,
																	 std::matrix<double>& gradients) = 0;

protected:
	Neuron* _neuron;
};

/**
 * Mean-squared error cost function
 */
class MSE : public CostFunc
{
public:
	MSE(Neuron* neuron)
		: CostFunc(neuron) {}

	virtual void calcOutputError(const std::vector<double>& labels,
															 const std::vector<double>& nnOutputs,
															 const std::vector<double>& sums,
															 std::vector<double>& errors);
	virtual void calcHiddenError(const std::matrix<double>& weights,
															 const std::vector<double>& nextErrors,
															 const std::vector<double>& sums,
															 std::vector<double>& errors);
	virtual void calcWeightGradients(const std::vector<double>& errors,
																	 const std::vector<double>& prevActivations,
																	 std::matrix<double>& gradients);
};

}

#endif /* _COST_FUNC_H */

