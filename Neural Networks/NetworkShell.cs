using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks
{
    abstract class LearningNetwork : Network, INetwork
    {
        public LearningNetwork(LearningType type, Functions func, params uint[] layers)
            : base(type, func, layers) { }
        public abstract void Learning(double[] args, double[] expectedRes);
    }

    class NetworkREP : LearningNetwork
    {
        public double LearningNorm;
        public double InertialTerm;
        public NetworkREP(LearningType type, Functions func, params uint[] layers)
            : base(type, func, layers) { }

        public override void Learning(double[] args, double[] expectedRes)
        {
            if ((args.Length != NeuronsNetwork[0].Length)
                || expectedRes.Length != NeuronsNetwork[NeuronsNetwork.Length - 1].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру (выходного слоя + ожидаемых результатов)");
            }

            GetResult(args);

            Outputter.Log("Обучение...");
            Task.Run(() =>
            {
                Parallel.ForEach(NeuronsNetwork[NeuronsNetwork.Length - 1], outputNeuron =>
                {
                    outputNeuron.RecieveErrorSigmoid(null, LearningNorm, InertialTerm,
                        expectedRes[(outputNeuron as OutputNeuron).NeuronId]);
                }
                  );
            }
            ).Wait();
            Outputter.Log("Обучение завершено.");

        }
    }


}
