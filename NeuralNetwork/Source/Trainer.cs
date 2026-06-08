using NeuralNetwork.Source.Data;

namespace NeuralNetwork.Source;

public class Trainer
{
    public readonly int Instances;
    public readonly int SurvivalRate;
    
    public double MutationRate = 0.05;
    public double MutationStrength = 0.1;

    public readonly int InputSize, OutputSize;
    
    private readonly Network[] _networks;
    private readonly double[] _scores;
    
    private readonly int[] layers;
    
    private static readonly ThreadLocal<Random> Random = 
        new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));
    
    public Trainer(int instances, int survivalRate, params int[] layers)
    {
        Instances = instances;
        SurvivalRate = survivalRate;
        
        InputSize = layers[0];
        OutputSize = layers[^1];
        
        if (instances <= 0)
            throw new ArgumentException("Instances must be positive", nameof(instances));
        if (survivalRate <= 0 || survivalRate > instances)
            throw new ArgumentException("Survival rate must be positive and less than or equal to instances", nameof(survivalRate));
        
        _networks = new Network[instances];
        _scores = new double[instances];
        for (int i = 0; i < instances; i++)
        {
            _networks[i] = new Network(0, layers);
            _scores[i] = 0;
        }
        
        this.layers = layers;
    }

    public void Run(int epochs, Func<Network, double> fitnessFunction)
    {
        for (int generation = 0; generation < epochs; generation++)
        {
            Console.Write($"\rTraining - {(generation + 1) * 100.0 / epochs:F2}% [{generation + 1}/{epochs}]");
        
            Parallel.For(0, Instances, i =>
                _scores[i] = fitnessFunction(_networks[i]));

            Array.Sort(_scores, _networks);
            Array.Reverse(_scores); Array.Reverse(_networks);

            Network[] survivors = _networks[..SurvivalRate];
            Network[] nextGen = new Network[Instances];

            for (int i = 0; i < SurvivalRate; i++)
                nextGen[i] = survivors[i];

            for (int i = SurvivalRate; i < Instances; i++)
            {
                Network child = new Network(0, layers);
                Network parentA = survivors[Random.Value.Next(SurvivalRate)];
                Network parentB = survivors[Random.Value.Next(SurvivalRate)];
                double[] paramsA = parentA.GetParameters();
                double[] paramsB = parentB.GetParameters();
                double[] childParams = new double[paramsA.Length];
                for (int j = 0; j < childParams.Length; j++)
                    childParams[j] = Random.Value.NextDouble() < 0.5 ? paramsA[j] : paramsB[j];
                for (int j = 0; j < childParams.Length; j++)
                    if (Random.Value.NextDouble() < MutationRate)
                        childParams[j] += Random.Value.NextGaussian() * MutationStrength;
                child.SetParameters(childParams);
                nextGen[i] = child;
            }

            Array.Copy(nextGen, _networks, Instances);
        }
    }
    
    public double[] Forward(double[] input) => _networks[0].Forward(input);
}