#include <cmath>
#include "Neuron.h"

using namespace ML;

///////////////////////////////////////////////////////////////////////////////
// Sigmoid neuron model implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * Activate according to the sigmoid function, i.e.,
 *
 *           1
 * y(z) = -------
 *             -z
 *        1 + e
 *
 * where z is the weighted sum from the previous layer for the current neuron
 */
double SigmoidNeuron::activate(double sum)
{
	return 1.0 / (1.0 + exp(sum * -1.0));
}

/**
 * Calculate gradient according to the derivative of the sigmoid function,
 * i.e.,
 *
 *             z
 *            e
 * y'(z) = ----------
 *               z  2
 *         (1 + e )
 *
 * where z is the weighted sum from the previous layer for the current neuron
 */
double SigmoidNeuron::gradient(double sum)
{
	return exp(sum) / pow(1.0 + exp(sum), 2.0);
}

///////////////////////////////////////////////////////////////////////////////
// Hyperbolic tangent neuron model implementation
///////////////////////////////////////////////////////////////////////////////
// TODO

/**
 * 
 */
double TanHNeuron::activate(double sum)
{
	return tanh(sum);
}

/*double TanHNeuron::gradient(double sum)
{

}*/

