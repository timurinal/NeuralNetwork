using NeuralNetwork.Source.Data;

namespace NeuralNetwork.Source;

public sealed class Network
{
    public readonly Layer[] Layers;
    public readonly int InputSize, OutputSize;

    private readonly int _batchSize;
    private readonly double[][][] _activations;  // [slot][layer][neuron]
    private readonly LayerGradients[][] _grads;  // [slot][layer]
    private readonly LayerGradients[] _avgGrads; // [layer]

    public Network(int batchSize, params int[] layers)
    {
        if (layers.Length < 2) throw new ArgumentException("Need at least an input and output size");

        _batchSize = batchSize;
        Layers = new Layer[layers.Length - 1];
        for (int i = 0; i < layers.Length - 1; i++)
        {
            bool isOutput = i == layers.Length - 2;
            Layers[i] = new Layer(
                layers[i], layers[i + 1],
                isOutput ? Maths.None : Maths.Tanh,
                isOutput ? Maths.NoneDerivative : Maths.TanhDerivative
            );
        }

        InputSize = layers[0];
        OutputSize = layers[^1];

        _activations = new double[batchSize][][];
        _grads = new LayerGradients[batchSize][];

        for (int s = 0; s < batchSize; s++)
        {
            _activations[s] = new double[Layers.Length + 1][];
            _activations[s][0] = new double[InputSize];
            for (int i = 0; i < Layers.Length; i++)
                _activations[s][i + 1] = new double[Layers[i].OutputLength];

            _grads[s] = new LayerGradients[Layers.Length];
            for (int i = 0; i < Layers.Length; i++)
                _grads[s][i] = new LayerGradients(Layers[i].InputLength, Layers[i].OutputLength);
        }

        _avgGrads = new LayerGradients[Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
            _avgGrads[i] = new LayerGradients(Layers[i].InputLength, Layers[i].OutputLength);
    }

    public double[] Forward(double[] input)
    {
        double[] current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }

    private void ComputeGradients(int slot, double[] input, double[] target)
    {
        var acts = _activations[slot];
        var grads = _grads[slot];

        Array.Copy(input, acts[0], InputSize);

        for (int i = 0; i < Layers.Length; i++)
            Layers[i].Forward(acts[i], acts[i + 1]);

        for (int j = 0; j < OutputSize; j++)
            grads[^1].Delta[j] = acts[^1][j] - target[j];

        for (int i = Layers.Length - 2; i >= 0; i--)
        {
            for (int j = 0; j < Layers[i].OutputLength; j++)
            {
                double err = 0;
                for (int k = 0; k < Layers[i + 1].OutputLength; k++)
                    err += Layers[i + 1].Weights[j, k] * grads[i + 1].Delta[k];
                grads[i].Delta[j] = Layers[i].Derivative(acts[i + 1][j]) * err;
            }
        }

        // Weight and bias gradients
        for (int i = 0; i < Layers.Length; i++)
        {
            for (int j = 0; j < Layers[i].InputLength; j++)
                for (int k = 0; k < Layers[i].OutputLength; k++)
                    grads[i].Weights[j, k] = acts[i][j] * grads[i].Delta[k];

            for (int k = 0; k < Layers[i].OutputLength; k++)
                grads[i].Biases[k] = grads[i].Delta[k];
        }
    }

    public void Train(double[][] inputs, double[][] targets, double learningRate = 0.001)
    {
        if (inputs.Length != _batchSize)
            throw new ArgumentException($"Expected batch size {_batchSize}, got {inputs.Length}");

        Parallel.For(0, _batchSize, slot => ComputeGradients(slot, inputs[slot], targets[slot]));

        double invBatch = 1.0 / _batchSize;
        for (int i = 0; i < Layers.Length; i++)
        {
            _avgGrads[i].Clear();
            for (int s = 0; s < _batchSize; s++)
            {
                for (int j = 0; j < Layers[i].InputLength; j++)
                    for (int k = 0; k < Layers[i].OutputLength; k++)
                        _avgGrads[i].Weights[j, k] += _grads[s][i].Weights[j, k] * invBatch;

                for (int k = 0; k < Layers[i].OutputLength; k++)
                    _avgGrads[i].Biases[k] += _grads[s][i].Biases[k] * invBatch;
            }

            Layers[i].ApplyGradients(_avgGrads[i].Weights, _avgGrads[i].Biases, learningRate);
        }
    }
    
    public void SetWeights(double[][,] weights)
    {
        if (weights.Length != Layers.Length)
            throw new ArgumentException($"Expected {Layers.Length} weights, got {weights.Length}");

        for (int i = 0; i < Layers.Length; i++)
            Layers[i].SetWeights(weights[i]);
    }

    public double[][,] GetWeights()
    {
        double[][,] result = new double[Layers.Length][,];
        for (int i = 0; i < Layers.Length; i++)
            result[i] = Layers[i].GetWeights();
        return result;
    }
    
    public double[] GetParameters()
    {
        int total = Layers.Sum(l => l.InputLength * l.OutputLength + l.OutputLength);
        double[] result = new double[total];
        int idx = 0;
        foreach (var layer in Layers)
        {
            for (int i = 0; i < layer.InputLength; i++)
            for (int j = 0; j < layer.OutputLength; j++)
                result[idx++] = layer.Weights[i, j];
            for (int j = 0; j < layer.OutputLength; j++)
                result[idx++] = layer.Biases[j];
        }
        return result;
    }

    public void SetParameters(double[] parameters)
    {
        int idx = 0;
        foreach (var layer in Layers)
        {
            for (int i = 0; i < layer.InputLength; i++)
            for (int j = 0; j < layer.OutputLength; j++)
                layer.Weights[i, j] = parameters[idx++];
            for (int j = 0; j < layer.OutputLength; j++)
                layer.Biases[j] = parameters[idx++];
        }
    }
}