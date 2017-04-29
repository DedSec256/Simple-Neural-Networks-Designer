using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Neural_Networks
{

    public static class Outputter
    {
        public static void Log(string text)
        {
            //Console.WriteLine(text);
        }

        public static void WriteLine(string text)
        {
            Console.WriteLine(text);
        }

        public static void Error(string text)
        {
            Console.WriteLine(text);
        }

        public static void Warning(string text)
        {
            Console.WriteLine("[Предупреждение] " + text);
        }
    }
    class Network
    {
        public enum LearningType
        {
            Reverse_error_propagation
        }

        LearningType NetworkLearningType;

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
        public bool IsConfigured
        {
            get; private set;
        }
        // TODO:  Предупреждение, что некоторый нейрон не используется
        public Network(params uint[] layers)
        {
            if (layers.Length <= 1)
                throw new ArgumentException("Сеть не может состоять меньше, чем из 2-х слоёв");

            NeuronsNetwork = new Neuron[layers.Length][];
            for (uint i = 0; i < layers.Length; i++)
            {
                if (layers[i] == 0) throw new ArgumentException($"Значение слоя {i} не может быть нулевым");
                NeuronsNetwork[i] = new Neuron[layers[i]];
            }
            for (uint i = 0; i < NeuronsNetwork[0].Length; i++)
            {
                NeuronsNetwork[0][i] = new InputNeuron($"Входной нейрон [{0}][{i}]", i);
            }
            for (uint i = 1; i < layers.Length - 1; i++)
            {
                for (uint j = 0; j < NeuronsNetwork[i].Length; j++)
                {
                    NeuronsNetwork[i][j] = new HiddenNeuron($"Скрытый нейрон [{i}][{j}]", i);
                }
            }
            for (uint i = 0; i < NeuronsNetwork[layers.Length - 1].Length; i++)
            {
                NeuronsNetwork[layers.Length - 1][i] = new OutputNeuron($"Выходной нейрон [{layers.Length - 1}][{i}]", i);
            }

        }
        public Network(LearningType type, Functions func, params uint[] layers) : this(layers)
        {
            NetworkLearningType = type;
            NetFunctions = func;
        }
        public void TryToConfigure()
        {
            /*TODO: Проверка выходных нейронов на входящие 
             *      Проверка скрытых нейронов на входящие и исходящие
             *      проверка входных на исходящие
             * */
        }
        protected void ClearNeurons()
        {
            foreach (Neuron[] layer in NeuronsNetwork)
            {
                foreach (Neuron neuron in layer)
                {
                    neuron.Clear();
                }
            }
        }
        public double[] GetResult(params double[] args)
        {
            ClearNeurons();

            if (args.Length != NeuronsNetwork[0].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру входного слоя.");
            }

            Outputter.Log($"Входные данные: ");
            for (int i = 0; i < args.Length; i++) Outputter.Log($"{NeuronsNetwork[0][i].NeuronName} = {args[i]}");
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
    }
}
