namespace NeuralNetwork.Source.Data;

public static class Maths
{
    public static double ReLU(double x) => x > 0 ? x : 0;
    public static double ReLUDerivative(double x) => x > 0 ? 1.0 : 0.0;
    
    public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
    public static double SigmoidDerivative(double x) => Sigmoid(x) * (1 - Sigmoid(x));
    
    public static double None(double x) => x;
    public static double NoneDerivative(double x) => 1.0;
    
    public static double Tanh(double x) => Math.Tanh(x);
    public static double TanhDerivative(double x) => 1.0 - x * x;
    
    public static double NextGaussian(this Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
    
    public static double RandRadians(this Random rng) => rng.NextDouble() * 2 * Math.PI;
}