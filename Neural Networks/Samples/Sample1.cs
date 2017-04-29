using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks.Samples
{
    class Sample1
    {
            static Network.ActivateFunction MyActivateFunc;
            static void Main(string[] args)
            {
                MyActivateFunc = delegate (double value)
                {
                    return value >= 0 ? 1 : 0;
                };

                try
                {
                    Network network = new Network(Network.LearningType.Reverse_error_propagation,
                                                  new Network.Functions(MyActivateFunc, null,null), 1, 1, 2, 4, 1, 1);

                    network[0, 0].AddNextNeuron(network[1, 1], -1);
                    network[0, 0].AddNextNeuron(network[1, 2], -1);

                    network[0, 1].AddNextNeuron(network[1, 1], -1);
                    network[0, 1].AddNextNeuron(network[1, 2], -1);

                    network[1, 0].AddNextNeuron(network[1, 1], 1.5);
                    network[1, 0].AsOffset();

                    network[1, 3].AddNextNeuron(network[1, 2], 0.5);
                    network[1, 3].AsOffset();

                    network[1, 1].AddNextNeuron(network[3, 0], 1);
                    network[1, 2].AddNextNeuron(network[3, 0], -1);

                    network[2, 0].AddNextNeuron(network[3, 0], -0.5);
                    network[2, 0].AsOffset();

                    Outputter.Log(network.ToString());

                    foreach (double res in network.GetResult(0, 0))
                    {
                        /* Выведет -0.5 => 0*/
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
