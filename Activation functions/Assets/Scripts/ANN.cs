using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
{
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha; //alpha is how much of prev inputs will effect the current input
    List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if (numHidden > 0)
        {
            layers.Add(new Layer(numNPerHidden, numInputs));

            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numOutputs, numNPerHidden));
        }

        else
        {
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    public List<double> Go(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);

        for (int i = 0; i < numHidden + 1; i++) //i is looping through layers
        {
            if (i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for (int j = 0; j < layers[i].numNeurons; j++) //j is looping through neurons in each layers
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++) //k is looping through inputs of each neuron
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }

                N -= layers[i].neurons[j].bias;

                if (i == numHidden)
                {
                    layers[i].neurons[j].output = ActivationFunctionOutput(N);
                }
                else
                {
                    layers[i].neurons[j].output = ActivationFunction(N);
                }

                outputs.Add(layers[i].neurons[j].output);
            }
        }

        UpdateWeights(outputs, desiredOutput);

        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        for (int i = numHidden; i >= 0; i--) //reverse loop = back propagation part
        {
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                if (i == numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    //calculations for how much a neuron is responsible for total error
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    double errorGradSum = 0;

                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }

                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    if (i == numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;

                    }
                }

                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }

    //yes multiple activation functions can be used together for better performance
    double ActivationFunctionOutput(double value)
    {
        return Sigmoid(value);
    }

    double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    //(aka binary step)
    double Step(double value) //type of activation function
    {
        if (value < 0) return 0;
        else return 1;
    }

    //(aka logistic softstep)
    double Sigmoid(double value) //type of activation function
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);

        //or

        //double k1 = 1 / (1 + System.Math.Exp(value));
        //return k1;
    }

    //it is an extension of sigmoid function
    //if you want -ve values in outputs use tanH
    //has vanishing Gradient problem i.e at a point neurons will not be trainable        
    double TanH(double value) //type of activation function
    {

        return (2 * (Sigmoid(2 * value)) - 1);

        //or

        //return ((double)System.Math.Exp(value) - (double)System.Math.Exp(-value)) / ((double)System.Math.Exp(value) + (double)System.Math.Exp(-value));
    }

    //relu is really good on hidden layers
    //has vanishing Gradient problem i.e at a point neurons will not be trainable for < 0 values
    double ReLu(double value) //type of activation function
    {
        if (value > 0) return value;
        else return 0;
    }

    double LeakyReLu(double value) //type of activation function
    {
        if (value > 0) return value;
        else return 0.001 * value;
    }
}
