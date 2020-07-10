using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    ANN ann;
    double sumSquareError = 0; //sumsq is how closely model fixes the data it is fed

    // Start is called before the first frame update
    void Start()
    {
        ann = new ANN(2, 1, 1, 2, 0.8);

        List<double> results;

        for(int i = 0; i < 200000; i++)
        {
            sumSquareError = 0;
            //XOR operation
            results = Train(0, 0, 0);
            sumSquareError += Mathf.Pow((float)results[0] - 0, 2);
            results = Train(0, 1, 1);
            sumSquareError += Mathf.Pow((float)results[0] - 1, 2);
            results = Train(1, 0, 1);
            sumSquareError += Mathf.Pow((float)results[0] - 1, 2);
            results = Train(1, 1, 0);
            sumSquareError += Mathf.Pow((float)results[0] - 0, 2);
        }
        Debug.Log("SSE: " + sumSquareError); //should be near to zero

        results = Train(0, 0, 0);
        Debug.Log("0 0 " + results[0]);
        results = Train(0, 1, 1);
        Debug.Log("0 1 " + results[0]);
        results = Train(1, 0, 1);
        Debug.Log("1 0 " + results[0]);
        results = Train(1, 1, 0);
        Debug.Log("1 1 " + results[0]);
    }

    List<double> Train(double i1, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);

        return (ann.Go(inputs, outputs));
    }
}
