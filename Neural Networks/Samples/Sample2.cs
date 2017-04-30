using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks.Samples
{
    class Sample2
    {
        static Network.ActivateFunction MyActivateFunc;
        static void Main(string[] args)
        {
            MyActivateFunc = delegate (double value)
            {
                return 1 / (1 + Math.Exp(-value));
            };

            try
            {
            NetworkREP network = new NetworkREP(new Network.Functions(MyActivateFunc, null, null), 2,4,1,1);
            network.LearningNorm = 0.25;
            network.InertialTerm = 1;
                network[0, 0].AddNextNeuron(network[1, 1], -0.2);
                network[0, 0].AddNextNeuron(network[1, 2], -0.1);

                network[0, 1].AddNextNeuron(network[1, 1], 0.1);
                network[0, 1].AddNextNeuron(network[1, 2], 0.3);

                network[1, 0].AddNextNeuron(network[1, 1], 0.1);
                network[1, 0].AsOffset();

                network[1, 3].AddNextNeuron(network[1, 2], 0.1);
                network[1, 3].AsOffset();

                network[1, 1].AddNextNeuron(network[3, 0], 0.2);
                network[1, 2].AddNextNeuron(network[3, 0], 0.3);

                network[2, 0].AddNextNeuron(network[3, 0], 0.2);
                network[2, 0].AsOffset();

            foreach (double res in network.GetResult(1, 0))
            {
                Outputter.Log("Ответ: " + res);
            }
            Console.ReadLine();
            Console.WriteLine();


            for (int i = 0; i < 10000; i++)
            {
                network.Learning(new double[] { 0, 1 }, new double[] { 0 });
                network.Learning(new double[] { 0, 0 }, new double[] { 0 });
                network.Learning(new double[] { 1, 0 }, new double[] { 0 });
                network.Learning(new double[] { 1, 1 }, new double[] { 1 });
            }
            Console.WriteLine("Done.");
            Console.ReadLine();

            

            foreach (double res in network.GetResult(1, 0))
            {
                Outputter.Log("Ответ: " + res);
            }
            Console.ReadLine();
            foreach (double res in network.GetResult(0, 1))
            {
                Outputter.Log("Ответ: " + res);
            }
            Console.ReadLine();
            foreach (double res in network.GetResult(0, 0))
            {
                Outputter.Log("Ответ: " + res);
            }
            Console.ReadLine();
            
            foreach (double res in network.GetResult(1,1))
            {
                Outputter.Log("Ответ: " + res);
            }
            


            }
            catch (Exception ex)
            {
                Outputter.Error(ex.Message);
            }

            Console.ReadLine();
        }
    }
}
