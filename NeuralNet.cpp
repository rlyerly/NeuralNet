#include <cstdlib>
#include <cmath>
#include <sstream>
#include <cassert>

#include "NeuralNet.h"

using namespace std;
using namespace ML;

#define MAX_RAND 1.0

///////////////////////////////////////////////////////////////////////////////
// Public API
///////////////////////////////////////////////////////////////////////////////

/**
 * Constructor - initializes the underlying neural network, preparing it for
 * approximation & training.
 *
 * @param layerDims a vector containing the layer dimensions of the neural
 * network, including the number of layers and the number of neurons per layer
 * @param stepSize when training, the step size used to update biases & weights
 * @param nntype the type of neural network, where SUPERVISED is a standard
 * neural network (unimplemented) & REINFORCEMENT_LEARNING is a neural network
 * that can be trained online
 * @param ntype the neuron model type (see Neuron.h for more details)
 * @param cf the type of cost function used to drive backpropagation (see
 * CostFunc.h for more details)
 */
NeuralNet::NeuralNet(const vector<size_t>& layerDims,
										 double stepSize,
										 NNType nntype,
										 NeuronType ntype,
										 CostFunction cf)
	: _nntype(nntype), _numLayers(layerDims.size()), _stepSize(stepSize),
		_layerDims(layerDims)
{
	assert(_nntype == REINFORCEMENT_LEARNING); // TODO
	assert(_numLayers > 1);

	// Allocate matrices & vectors for NN
	_weights = new matrix<double>[_numLayers-1];
	_weightGradients = new matrix<double>[_numLayers-1];
	_biases = new vector<double>[_numLayers-1];
	_sums = new vector<double>[_numLayers-1];
	_activations = new vector<double>[_numLayers-1];
	_errors = new vector<double>[_numLayers-1];

	for(size_t i = 0; i < _numLayers-1; i++)
	{
		_weights[i].resize(layerDims[i], layerDims[i+1]);
		_weightGradients[i].resize(layerDims[i], layerDims[i+1]);
		_biases[i].resize(layerDims[i+1]);
		_sums[i].resize(layerDims[i+1]);
		_activations[i].resize(layerDims[i+1]);
		_errors[i].resize(layerDims[i+1]);
	}

	// Initialize weights & biases
	srand(time(NULL));
	randomizeWeights(MAX_RAND);
	randomizeBiases(MAX_RAND);

	// Initialize neuron model for activation & backpropagation
	switch(ntype)
	{
	case SIGMOID:
		_neuron = new SigmoidNeuron();
		break;
/*	case TANH: TODO
		_neuron = new TanHNeuron();
		break;*/
	default:
		assert(false && "Unknown neuron model!");
		break;
	}

	// Initialize cost function for backpropagation
	switch(cf)
	{
	case MSE:
		_cf = new ML::MSE(_neuron); // g++ error if ML namespace isn't declared...?
		break;
	default:
		assert(false && "Unknown cost function!");
		break;
	}
}

/**
 * Destructor - works as advertised
 */
NeuralNet::~NeuralNet()
{
	if(_weights)
		delete [] _weights;
	if(_weightGradients)
		delete [] _weightGradients;
	if(_biases)
		delete [] _biases;
	if(_sums)
		delete [] _sums;
	if(_activations)
		delete [] _activations;
	if(_errors)
		delete [] _errors;
	if(_neuron)
		delete _neuron;
	if(_cf)
		delete _cf;
}

/**
 * Using an input state vector, approximate a state-action value
 */
void NeuralNet::approximate(const std::vector<double>& inputs,
														std::vector<double>& outputs)
{
	assert(_nntype == REINFORCEMENT_LEARNING);
	feedforward(inputs, outputs);
}

/**
 * Train using the reward given by the environment
 */
void NeuralNet::train(unsigned long action, double reward)
{
	assert(_nntype == REINFORCEMENT_LEARNING);
	assert(action < _activations[_numLayers-2].size());

	vector<double> outputs(_activations[_numLayers-2].size());
	for(size_t i = 0; i < outputs.size(); i++)
	{
		if(i == action)
			outputs[i] = reward;
		else
			outputs[i] = _activations[_numLayers-2][i];
	}
	feedback(outputs);
}

/**
 * Save the neural network in a Python-parsable format
 *
 * TODO save other parameters (NN type, num layers, step size, layer
 * dimensions, neuron model & cost function)
 *
 * Output is a tuple that contains the following fields:
 *   [0] List of weight matrices
 *   [1] List of bias vectors
 */
ostream& NeuralNet::save(ostream& out) const
{
	string data("([");
	stringstream ss;

	// Weight matrices
	for(size_t i = 0; i < _numLayers - 1; i++)
		ss << _weights[i] << ",";
	data += ss.str();
	data.resize(data.length() - 1);
	data += "],[";

	// Bias vectors
	ss.str(string());
	for(size_t i = 0; i < _numLayers - 1; i++)
		ss << _biases[i] << ",";
	data += ss.str();
	data.resize(data.length() - 1);
	data += "])";

	out << data;

	// TODO remove
	out << endl;
	for(size_t i = 0; i < _numLayers - 1; i++)
		out << _sums[i] << endl;

	return out;
}

istream& NeuralNet::restore(istream& in)
{
//TODO
	return in;
}

///////////////////////////////////////////////////////////////////////////////
// Private API
///////////////////////////////////////////////////////////////////////////////

/**
 * Feed inputs forward through the network to generate outputs, saves
 * appropriate intermediate results for backpropagation.
 *
 * TODO move matrix multiplications/vector additions to functions
 * TODO optimize (SIMD/cache blocking/FMADD)
 */
void NeuralNet::feedforward(const vector<double>& inputs,
														vector<double>& outputs)
{
	assert(inputs.size() == _weights[0].rows());
	assert(outputs.size() == _weights[_numLayers-2].cols());

	// Save inputs for backpropagation
	_inputs = inputs;

	// Initialize output vector
	for(size_t i = 0; i < outputs.size(); i++)
		outputs[i] = 0.0;

	// Feed forward
	if(_numLayers == 2)
	{
		for(size_t i = 0; i < _weights[0].rows(); i++)
			for(size_t j = 0; j < _weights[0].cols(); j++)
				outputs[j] += inputs[i] * _weights[0](i, j);
		for(size_t i = 0; i < outputs.size(); i++)
		{
			outputs[i] += _biases[0][i];
			_sums[0][i] = outputs[i];
			outputs[i] = _neuron->activate(outputs[i]);
			_activations[0][i] = outputs[i];
		}
	}
	else
	{
		size_t l = 0;
		zeroSums();

		// Input layer
		for(size_t i = 0; i < _weights[l].rows(); i++)
			for(size_t j = 0; j < _weights[l].cols(); j++)
				_sums[l][j] += inputs[i] * _weights[l](i, j);
		for(size_t i = 0; i < _activations[l].size(); i++)
		{
			_sums[l][i] += _biases[l][i];
			_activations[l][i] = _neuron->activate(_sums[l][i]);
		}

		// Hidden layers
		for(l++; l < _numLayers-2; l++)
		{
			for(size_t i = 0; i < _weights[l].rows(); i++)
				for(size_t j = 0; j < _weights[l].cols(); j++)
					_sums[l][j] += _activations[l-1][i] * _weights[l](i, j);
			for(size_t i = 0; i < _activations[l].size(); i++)
			{
				_sums[l][i] += _biases[l][i];
				_activations[l][i] = _neuron->activate(_sums[l][i]);
			}
		}

		// Output layer
		for(size_t i = 0; i < _weights[l].rows(); i++)
			for(size_t j = 0; j < _weights[l].cols(); j++)
				outputs[j] += _activations[l-1][i] * _weights[l](i, j);
		for(size_t i = 0; i < outputs.size(); i++)
		{
			outputs[i] += _biases[l][i];
			_sums[l][i] = outputs[i];
			outputs[i] = _neuron->activate(outputs[i]);
			_activations[l][i] = outputs[i];
		}
	}
}

/**
 * Core of backpropagation algorithm - calculate weight & bias gradients
 * by propagating errors back through the network.  On the way, update weights
 * & biases according to the step size.
 *
 * @param labels a vector containing the expected outputs or lables for the
 * previous inputs
 */
void NeuralNet::feedback(const vector<double>& labels)
{
	// Sanity checks
	assert(_errors[_numLayers-2].size() == labels.size());

	// Calculate errors, backpropagate & train
	if(_numLayers == 2)
	{
		_cf->calcOutputError(labels, _activations[0], _sums[0], _errors[0]);
		_cf->calcWeightGradients(_errors[0], _inputs, _weightGradients[0]);
	}
	else
	{
		size_t l = _numLayers - 2;

		// Output layer
		_cf->calcOutputError(labels, _activations[l], _sums[l], _errors[l]);
		_cf->calcWeightGradients(_errors[l], _activations[l-1], _weightGradients[l]);

		// Hidden layers
		for(l--; l > 0; l--)
		{
			_cf->calcHiddenError(_weights[l+1], _errors[l+1], _sums[l], _errors[l]);
			_cf->calcWeightGradients(_errors[l], _activations[l-1], _weightGradients[l]);
		}

		// Input layer
		_cf->calcHiddenError(_weights[1], _errors[1], _sums[0], _errors[0]);
		_cf->calcWeightGradients(_errors[0], _inputs, _weightGradients[0]);
	}

	// Update weights & biases after all gradients have been calculated
	for(size_t i = 0; i < _numLayers - 1; i++)
	{
		updateBiases(i);
		updateWeights(i);
	}
}

void NeuralNet::zeroWeights()
{
	for(size_t i = 0; i < _numLayers - 1; i++)
		for(size_t j = 0; j < _weights[i].rows(); j++)
			for(size_t k = 0; k < _weights[i].cols(); k++)
				_weights[i](j, k) = 0.0;
}

void NeuralNet::zeroBiases()
{
	for(size_t i = 0; i < _numLayers - 1; i++)
		for(size_t j = 0; j < _biases[i].size(); j++)
			_biases[i][j] = 0.0;
}

void NeuralNet::zeroSums()
{
	for(size_t i = 0; i < _numLayers - 1; i++)
		for(size_t j = 0; j < _sums[i].size(); j++)
			_sums[i][j] = 0.0;
}

void NeuralNet::randomizeWeights(double max)
{
	for(size_t i = 0; i < _numLayers - 1; i++)
		for(size_t j = 0; j < _weights[i].rows(); j++)
			for(size_t k = 0; k < _weights[i].cols(); k++)
				_weights[i](j, k) = ((double)rand() / RAND_MAX) * max;
}

void NeuralNet::randomizeBiases(double max)
{
	for(size_t i = 0; i < _numLayers - 1; i++)
		for(size_t j = 0; j < _biases[i].size(); j++)
			_biases[i][j] = ((double)rand() / RAND_MAX) * max;
}

void NeuralNet::updateBiases(size_t layer)
{
	for(size_t i = 0; i < _biases[layer].size(); i++)
		_biases[layer][i] -= _stepSize * _errors[layer][i];
}

void NeuralNet::updateWeights(size_t layer)
{
	for(size_t i = 0; i < _weights[layer].rows(); i++)
		for(size_t j = 0; j < _weights[layer].cols(); j++)
			_weights[layer](i, j) -= _stepSize * _weightGradients[layer](i, j);
}

