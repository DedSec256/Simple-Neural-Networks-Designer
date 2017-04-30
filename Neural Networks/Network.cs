using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Neural_Networks
{
    public class Network
    {
        public struct Functions
        {
            public readonly ActivateFunction ActivateFunc;
            public readonly OutErrorFunction OutErrorFunc;
            public readonly HiddenLayerErrorFunction HiddenLayerErrorFunc;
            public Functions(ActivateFunction func, OutErrorFunction outError, HiddenLayerErrorFunction hiddenError)
            {
                ActivateFunc = func;
                OutErrorFunc = outError;
                HiddenLayerErrorFunc = hiddenError;
            }
        }
        public delegate double ActivateFunction(double value);
        public delegate double OutErrorFunction(double value, double expectedOutput);
        public delegate double HiddenLayerErrorFunction(double value);

        protected Neuron[][] NeuronsNetwork;
        public Functions NetFunctions;
        public Network(params uint[] layers)
        {
            if (layers.Length <= 1)
                throw new ArgumentException("Сеть не может состоять меньше, чем из 2-х слоёв");

            NeuronsNetwork = new Neuron[layers.Length][];
            Task.Run(() =>  
            {
                Parallel.For(0, layers.Length, i =>
                {
                    if (layers[i] == 0) throw new ArgumentException($"Значение слоя {i} не может быть нулевым");
                    NeuronsNetwork[i] = new Neuron[layers[i]];
                });
            }).Wait();

            Parallel.For(0, NeuronsNetwork[0].Length, i =>
            {
                NeuronsNetwork[0][i] = new InputNeuron($"Входной нейрон [{0}][{i}]", Convert.ToUInt32(i));
            });

            Parallel.For(1, layers.Length - 1, i =>
            {
                Parallel.For(0, NeuronsNetwork[i].Length, j =>
                {
                    NeuronsNetwork[i][j] = new HiddenNeuron($"Скрытый нейрон [{i}][{j}]", Convert.ToUInt32(i));
                });
            });

            Parallel.For(0, NeuronsNetwork[layers.Length - 1].Length, i =>
            {
                NeuronsNetwork[layers.Length - 1][i] = new OutputNeuron($"Выходной нейрон [{layers.Length - 1}][{i}]", Convert.ToUInt32(i));
            });

        }
        public Network(Functions func, params uint[] layers) : this(layers)
        {
            WriteLog = false;
            NetFunctions = func;
        }
        public void TryToConfigure()
        {
            // TODO:  Предупреждение, что некоторый нейрон не используется
            /*TODO: Проверка выходных нейронов на входящие 
             *      Проверка скрытых нейронов на входящие и исходящие
             *      проверка входных на исходящие
             * */
        }
        protected void ClearNeurons()
        {
            Parallel.ForEach(NeuronsNetwork, layer => {
                Parallel.ForEach(layer, neuron =>
                {
                    neuron.Clear();
                }
                );
            });
        }
        public double[] GetResult(params double[] args)
        {
            ClearNeurons();

            if (args.Length != NeuronsNetwork[0].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру входного слоя.");
            }

            if(WriteLog) Outputter.Log($"Входные данные: ");

            for (int i = 0; i < args.Length; i++)
                if (WriteLog) Outputter.Log($"{NeuronsNetwork[0][i].NeuronName} = {args[i]}");
            Task.Run(() =>
                {
                    Parallel.ForEach(NeuronsNetwork[0], inputNeuron =>
                    {
                        inputNeuron.RecieveSignal(args[(inputNeuron as InputNeuron).NeuronId], NetFunctions.ActivateFunc);
                    }
                    );
                }
            ).Wait();

            Task.Run(() =>
            {
                for (int i = 1; i < NeuronsNetwork.Length - 1; i++)
                {
                    Parallel.ForEach(NeuronsNetwork[i].Where(t => t.GetRemainingSignals() == 0), hiddenNeuron =>
                    {
                        hiddenNeuron.SendSignal(NetFunctions.ActivateFunc);
                    }
                    );
                }
            }
            ).Wait();

            double[] result = new double[NeuronsNetwork[NeuronsNetwork.Length - 1].Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NeuronsNetwork[NeuronsNetwork.Length - 1][i].GetValue();
            }
            return result;
        }
        public Neuron this[int i, int j]
        {
            get
            {
                return NeuronsNetwork[i][j];
            }
            private set
            {
                NeuronsNetwork[i][j] = value;
            }
        }
        public int LayersCount
        {
            get { return NeuronsNetwork.Length; }
        }
        public Neuron[] this[int i]
        {
            get
            {
                return NeuronsNetwork[i];
            }
            private set
            {
                NeuronsNetwork[i] = value;
            }
        }
        public override string ToString()
        {
            StringBuilder sB = new StringBuilder();
            sB.AppendLine("Описание нейросети:");
           for(int i = 0; i < NeuronsNetwork.Length; i++)
            {
                sB.AppendLine($"[Слой {i}]");
                for(int j = 0; j< NeuronsNetwork[i].Length; j++)
                {
                    sB.Append("-" + NeuronsNetwork[i][j].ToString());
                }
                sB.AppendLine();
            }
            return sB.ToString();
        }

        public bool WriteLog;
    }
}
