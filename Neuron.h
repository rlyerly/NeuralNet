/**
 * Neuron models used for activation & differentiation.
 *
 * Author: Rob Lyerly
 * Date: 9/1/2014
 */

#ifndef _NEURON_H
#define _NEURON_H

namespace ML
{

/**
 * Neuron model super-class which describes the interface required for
 * different neuron models
 */
class Neuron
{
public:
	/* Appease the whiny compiler */
	virtual ~Neuron() {};

	/**
	 * Apply activation function to weighted sum from previous layer
	 */
	virtual double activate(double sum) = 0;

	/**
	 * Calculate rate of change of activation function from the weighted sum from
	 * previous layer
	 */
	virtual double gradient(double sum) = 0;
};

/**
 * Classic sigmoid neuron model.  Activation function follows a differentiable
 * curve from 0 <= y(x) <= 1
 */
class SigmoidNeuron : public Neuron
{
public:
	virtual double activate(double sum);
	virtual double gradient(double sum);
};

/**
 * Hyperbolic-tangent neuron model.  Similar to sigmoidal, but activation
 * function is a differentiable curve from -1 <= y(x) <= 1
 */
class TanHNeuron : public Neuron
{
public:
	virtual double activate(double sum);
//	virtual double gradient(double sum);
};

}

#endif /* _NEURON_H */

