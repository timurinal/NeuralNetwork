using NeuralNetwork.Source;
using NeuralNetwork.Source.Data;

namespace NeuralNetwork;

class Program
{
    static void Main(string[] args)
    {
        const int instances   = 100;
        const int survivalRate = 1;
        const int epochs      = 1000;

        var trainer = new Trainer(instances, survivalRate, 1, 16, 16, 1);
        var rng = new ThreadLocal<Random>(() => new Random(Guid.NewGuid().GetHashCode()));

        var timer = System.Diagnostics.Stopwatch.StartNew();

        trainer.Run(epochs, network =>
        {
            double totalError = 0;
            const int samples = 50;
            for (int i = 0; i < samples; i++)
            {
                double x = rng.Value!.RandRadians();
                double predicted = network.Forward([x])[0];
                double actual = Math.Sin(x) * Math.Cos(2 * x);
                totalError += Math.Abs(predicted - actual);
            }
            return -totalError / samples; // negative error = higher is better
        });

        timer.Stop();
        Console.WriteLine($"\nTraining complete in {FormatDuration(timer.ElapsedMilliseconds)}");

        double[] testPoints = [0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI];
        foreach (double x in testPoints)
        {
            double predicted = trainer.Forward([x])[0];
            double actual    = Math.Sin(x) * Math.Cos(2 * x);
            double accuracy  = Math.Max(0, 1.0 - Math.Abs(predicted - actual)) * 100.0;
            Console.WriteLine($"f({x:F2}) → predicted: {predicted:F4}, actual: {actual:F4} (Accuracy: {accuracy:F2}%)");
        }
    }
    
    static string FormatDuration(double milliseconds)
    {
        var ts = TimeSpan.FromMilliseconds(milliseconds);
        if (ts.TotalHours >= 1)
            return $"{(int)ts.TotalHours}h {ts.Minutes}m {ts.Seconds}s";
        if (ts.TotalMinutes >= 1)
            return $"{ts.Minutes}m {ts.Seconds}s";
        return $"{ts.TotalSeconds:F1}s";
    }
}