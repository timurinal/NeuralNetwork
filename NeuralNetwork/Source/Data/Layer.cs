namespace NeuralNetwork.Source.Data;

public sealed class Layer
{
    public readonly double[,] Weights;
    public readonly double[] Biases;
    public readonly int InputLength, OutputLength;

    private readonly Func<double, double> _activation;
    private readonly Func<double, double> _activationDerivative;

    // Adam state
    private readonly double[,] _mW, _vW;
    private readonly double[] _mB, _vB;
    private int _t;
    private const double Beta1 = 0.9, Beta2 = 0.999, Eps = 1e-8;

    public Layer(int inputSize, int outputSize, Func<double, double> activation, Func<double, double> activationDerivative)
    {
        Weights = new double[inputSize, outputSize];
        Biases = new double[outputSize];
        InputLength = inputSize;
        OutputLength = outputSize;
        _activation = activation;
        _activationDerivative = activationDerivative;

        _mW = new double[inputSize, outputSize];
        _vW = new double[inputSize, outputSize];
        _mB = new double[outputSize];
        _vB = new double[outputSize];

        var rng = new Random();
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < outputSize; j++)
                Weights[i, j] = rng.NextGaussian() * Math.Sqrt(1.0 / inputSize); // Xavier
    }

    public void Forward(double[] input, double[] output)
    {
        for (int j = 0; j < OutputLength; j++)
        {
            double sum = Biases[j];
            for (int i = 0; i < InputLength; i++)
                sum += Weights[i, j] * input[i];
            output[j] = _activation(sum);
        }
    }

    public double[] Forward(double[] input)
    {
        var output = new double[OutputLength];
        Forward(input, output);
        return output;
    }

    public double Derivative(double activationOutput) => _activationDerivative(activationOutput);

    public void ApplyGradients(double[,] weightGrads, double[] biasGrads, double lr)
    {
        _t++;
        double bc1 = 1 - Math.Pow(Beta1, _t);
        double bc2 = 1 - Math.Pow(Beta2, _t);

        for (int i = 0; i < InputLength; i++)
        {
            for (int j = 0; j < OutputLength; j++)
            {
                double g = weightGrads[i, j];
                _mW[i, j] = Beta1 * _mW[i, j] + (1 - Beta1) * g;
                _vW[i, j] = Beta2 * _vW[i, j] + (1 - Beta2) * g * g;
                Weights[i, j] -= lr * (_mW[i, j] / bc1) / (Math.Sqrt(_vW[i, j] / bc2) + Eps);
            }
        }

        for (int j = 0; j < OutputLength; j++)
        {
            double g = biasGrads[j];
            _mB[j] = Beta1 * _mB[j] + (1 - Beta1) * g;
            _vB[j] = Beta2 * _vB[j] + (1 - Beta2) * g * g;
            Biases[j] -= lr * (_mB[j] / bc1) / (Math.Sqrt(_vB[j] / bc2) + Eps);
        }
    }
    
    public void SetWeights(double[,] weights)
    {
        Array.Copy(weights, Weights, weights.Length);
    }
    
    public double[,] GetWeights() => (double[,])Weights.Clone();
}

public sealed class LayerGradients
{
    public readonly double[,] Weights;
    public readonly double[] Biases;
    public readonly double[] Delta;

    public LayerGradients(int inputSize, int outputSize)
    {
        Weights = new double[inputSize, outputSize];
        Biases = new double[outputSize];
        Delta = new double[outputSize];
    }

    public void Clear()
    {
        Array.Clear(Weights);
        Array.Clear(Biases);
    }
}