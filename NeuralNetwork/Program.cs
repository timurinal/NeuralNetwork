using NeuralNetwork.Source;
using NeuralNetwork.Source.Data;

namespace NeuralNetwork;

class Program
{
    static void Main(string[] args)
    {
        const int batchSize = 8192;
        const int epochs = 10000;
        
        var nn = new Network(batchSize, 1, 16, 16, 1);

        var rng = new Random();

        var timer = new System.Diagnostics.Stopwatch();
        timer.Start();

        var inputs = new double[batchSize][];
        var targets = new double[batchSize][];
        for (int i = 0; i < batchSize; i++)
        {
            inputs[i] = new double[1];
            targets[i] = new double[1];
        }

        for (int step = 0; step < epochs; step++)
        {
            //if (step % 10 == 0)
            {
                long elapsedMs = timer.ElapsedMilliseconds;
                int completed = step + 1;
                double avgMsPerStep = (double)elapsedMs / completed;
                double etaMs = avgMsPerStep * (epochs - completed);

                Console.Write($"\rTraining - {completed * 100.0 / epochs:F2}% - " +
                              $"ETA: {FormatDuration(etaMs)} " +
                              $"[{completed * batchSize}/{epochs * batchSize}]");
            }

            for (int i = 0; i < batchSize; i++)
            {
                inputs[i][0] = rng.RandRadians();
                targets[i][0] = Math.Sin(inputs[i][0]);
            }
            nn.Train(inputs, targets);
        }
        timer.Stop();
        Console.WriteLine($"\nTraining complete in {FormatDuration(timer.ElapsedMilliseconds)}");

        double[] testPoints = [0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI];
        foreach (double x in testPoints)
        {
            double predicted = nn.Forward([x])[0];
            double actual = Math.Sin(x);

            double error = Math.Abs(predicted - actual);
            double accuracy = Math.Max(0, 1.0 - error) * 100.0;

            Console.WriteLine($"sin({x:F2}) → predicted: {predicted:F4}, actual: {actual:F4} (Accuracy: {accuracy:F2}%)");
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