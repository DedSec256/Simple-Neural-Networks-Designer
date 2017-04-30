using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks
{
   public abstract class LearningNetwork : Network, INetwork
    {
        public LearningNetwork(Functions func, params uint[] layers)
            : base(func, layers) { }
        public abstract void Learning(double[] args, double[] expectedRes);
    }

    public class NetworkREP : LearningNetwork
    {
        public double LearningNorm;
        public double InertialTerm;
        public NetworkREP(Functions func, params uint[] layers)
            : base(func, layers) { }

        public override void Learning(double[] args, double[] expectedRes)
        {
            if ((args.Length != NeuronsNetwork[0].Length)
                || expectedRes.Length != NeuronsNetwork[NeuronsNetwork.Length - 1].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру (выходного слоя + ожидаемых результатов)");
            }

            GetResult(args);

            if(WriteLog) Outputter.Log("Обучение...");
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
            if (WriteLog) Outputter.Log("Обучение завершено.");

        }
    }


}
