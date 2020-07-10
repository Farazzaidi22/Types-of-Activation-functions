using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    public int numNeurons;
    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int nNeuron, int numNeuronInputs)
    {
        numNeurons = nNeuron;
        for(int i = 0; i< nNeuron; i++)
        {
            neurons.Add(new Neuron(numNeuronInputs));
        }
    }
}
