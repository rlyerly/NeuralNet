/**
 * NeuralNet - a C++ neural network implementation.
 *
 * Based on the book "Neural Networks and Deep Learning" by Michael Nielson
 * (http://neuralnetworksanddeeplearning.com)
 *
 * Author: Rob Lyerly
 * Date: 8/29/2014
 */

#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H

#include <iostream>

#include "matrix.h"
#include "vector.h"

/*
 * Network neuron models & cost functions
 */
#include "Neuron.h"
#include "CostFunc.h"

namespace ML
{

class NeuralNet
{
public:
	/**
	 * Types of neural network
	 *
	 * Supervised - TODO
	 *
	 * Reinforcement learning - train the network one observation at a time.
	 * Requires a single output label (from the environment).
	 */
	typedef enum NNType {
		SUPERVISED,
		REINFORCEMENT_LEARNING
	} NNType;

	/**
	 * Constructors & destructors
	 */
	NeuralNet(const std::vector<size_t>& layerDims,
						double stepSize,
						NNType nntype,
						Neuron::NeuronType ntype = Neuron::SIGMOID,
						CostFunc::CostFuncType cftype = CostFunc::MSE);
	~NeuralNet();

	/**
	 * Basic neural network functionality for reinforcement learning
	 */
	void approximate(const std::vector<double>& inputs,
									 std::vector<double>& outputs);
	void train(unsigned long action, double reward);

	/**
	 * Save/restore neural network & corresponding convenience functions
	 */
	std::ostream& save(std::ostream& out) const;
	std::istream& restore(std::istream& in);
	friend std::ostream& operator<<(std::ostream& out, const NeuralNet& nn)
		{ return nn.save(out); }
	friend std::istream& operator>>(std::istream& in, NeuralNet& nn)
		{ return nn.restore(in); }

	/**
	 * Getters & setters
	 */
	void setStepSize(double stepSize) { _stepSize = stepSize; }
	double getStepSize(double stepSize) { return _stepSize; }

private:
	/**
	 * General network configuration
	 */
	NNType _nntype;
	size_t _numLayers;
	double _stepSize;
	std::vector<size_t> _layerDims;
	Neuron* _neuron;
	CostFunc* _cf;

	/**
	 * Underlying network, maintained as a set of matrices & vectors
	 */
	std::matrix<double>* _weights;
	std::matrix<double>* _weightGradients;
	std::vector<double>* _biases;
	std::vector<double>* _sums;
	std::vector<double>* _activations;
	std::vector<double>* _errors;
	std::vector<double> _inputs;

	/**
	 * Private helper functions
	 */
	void feedforward(const std::vector<double>& inputs,
									 std::vector<double>& outputs);
	void feedback(const std::vector<double>& labels);
	void zeroWeights();
	void zeroBiases();
	void zeroSums();
	void randomizeWeights(double max);
	void randomizeBiases(double max);
	void updateBiases(size_t layer);
	void updateWeights(size_t layer);
};

}

#endif /* _NEURAL_NET_H */

