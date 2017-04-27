using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Neural_Networks
{
    abstract class Neuron
    {
        public enum Types
        {
            Input, Hidden, Output
        }
        public Types Type
        {
            get; private set;
        }

        public string NeuronName;
        protected Dictionary<Neuron, double> NextNeurons;
        protected Dictionary<Neuron, double> LastNeurons;
        protected int MaxAdmissionsleft;
        protected int AdmissionsLeft;
        protected double Value;
        public double GetValue() { return Value; }

        public virtual void AddNextNeuron(Neuron neuron, double weight)
        {
            if (NextNeurons.ContainsKey(neuron)) throw new
                    InvalidOperationException($"Нейроны {NeuronName} и {neuron.NeuronName} уже обьединены прямой связью.");
            else
            {
                NextNeurons.Add(neuron, weight);
                neuron.AddLastNeuron(this, weight);
                neuron.AdmissionsLeft = ++neuron.MaxAdmissionsleft;
            }
        }
        public virtual void AddLastNeuron(Neuron neuron, double weight)
        {
            if (LastNeurons.ContainsKey(neuron)) throw new
                    InvalidOperationException($"Нейроны {NeuronName} и {neuron.NeuronName} уже обьединены обратной связью.");
            else
            {
                LastNeurons.Add(neuron, weight);
            }
        }

        public virtual void RecieveSignal(double value, Network.ActivateFunction activFunc)
        {
            Interlocked.Exchange(ref Value, Value + value);
            Interlocked.Decrement(ref AdmissionsLeft);
            if (AdmissionsLeft == 0) SendSignal(activFunc);
        }

        private object sendSignalLock = new object();
        //TODO: --AdmissionLeft
        public virtual void SendSignal(Network.ActivateFunction activFunc)
        {
            Parallel.ForEach(NextNeurons, nextNeuron =>
            {
                nextNeuron.Key.RecieveSignal(activFunc(nextNeuron.Value * Value), activFunc);
            }
            );
        }
        public Neuron(string neuronName, Types type)
        {
            NeuronName = neuronName;
            Type = type;

            MaxAdmissionsleft = AdmissionsLeft = 0;
            Value = 0;

            NextNeurons = new Dictionary<Neuron, double>();
            LastNeurons = new Dictionary<Neuron, double>();
        }
        public override string ToString()
        {
            return NeuronName;
        }
    }

    class InputNeuron: Neuron
    {
        public InputNeuron(string neuronName) : base(neuronName, Types.Input)
        {
            NextNeurons = new Dictionary<Neuron, double>();
            MaxAdmissionsleft = AdmissionsLeft = 1;
        }
        public override void AddLastNeuron(Neuron neuron, double weight)
        {
            throw new Exception($"Нельзя соединить нейроны {neuron.NeuronName} и {NeuronName} обратной связью.");
        }

    }
    class HiddenNeuron : Neuron
    {
        uint HiddenLayerId;
        bool IsOffset;
        public HiddenNeuron(string neuronName, uint hiddenLayerId, bool isOffset = false) 
            : base(neuronName, Types.Hidden)
        {
            HiddenLayerId = hiddenLayerId;
            IsOffset = isOffset;
        }
    }
    class OutputNeuron : Neuron
    {
        public OutputNeuron(string neuronName) : base(neuronName, Types.Output)
        {
            LastNeurons = new Dictionary<Neuron, double>();
        }

        public override void AddNextNeuron(Neuron neuron, double weight)
        {
            throw new Exception($"Нельзя соединить нейроны {neuron.NeuronName} и {NeuronName} прямой связью.");
        }
    }
    class Network
    {
        public struct Functions
        {
            public readonly ActivateFunction ActivateFunc;
            public Functions(ActivateFunction func)
            {
                ActivateFunc = func;
            }
        }
        public delegate double ActivateFunction(double value);

        Neuron[][] NeuronsNetwork;
        public Functions NetFunctions;
        public bool IsConfigured
        {
            get; private set;
        }
        // TODO:  Предупреждение, что некоторый нейрон не используется
        public Network(params uint[] layers)
        {
            if (layers.Length <= 1)
                throw new RankException("Сеть не может состоять меньше, чем из 2-х слоёв");

            NeuronsNetwork = new Neuron[layers.Length][];
            for(uint i = 0; i<layers.Length; i++)
            {
                if (layers[i] == 0) throw new ArgumentException($"Значение слоя {i} не может быть нулевым");
                NeuronsNetwork[i] = new Neuron[layers[i]];
            }
            for(uint i = 0; i< NeuronsNetwork[0].Length; i++)
            {
                NeuronsNetwork[0][i] = new InputNeuron($"Входной нейрон [{0}][{i}]");
            }
            for(uint i=1; i<layers.Length - 1; i++)
            {
                for(uint j = 0; j< NeuronsNetwork[i].Length; j++)
                {
                    NeuronsNetwork[i][j] = new HiddenNeuron($"Скрытый нейрон [{i}][{j}]", i);
                }
            }
            for (uint i = 0; i < NeuronsNetwork[layers.Length - 1].Length; i++)
            {
                NeuronsNetwork[layers.Length - 1][i] = new OutputNeuron($"Выходной нейрон [{layers.Length - 1}][{i}]");
            }

        }
        public Network(Functions func, params uint[] layers) : this(layers)
        {
            NetFunctions = func;
        }
        public void Configure()
        {

        }

        public double[] GetResult(params double[] args)
        {
            if(args.Length != NeuronsNetwork[0].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру входного слоя.");
            }
            //TODO: Recieve для output
                          //для input
            Task.Run(() => 
                {
                    int index = 0;
                    Parallel.ForEach(NeuronsNetwork[0], inputNeuron =>
                    {
                        inputNeuron.RecieveSignal(args[index]);
                        ++index;
                        inputNeuron.SendSignal(delegate (double value) { return value; });
                    }
                    );
                }
            ).Wait();

            double[] result = new double[NeuronsNetwork[NeuronsNetwork.Length - 1].Length];
            for (int i = 0; i< result.Length; i++)
            {
                result[i] = NeuronsNetwork[NeuronsNetwork.Length - 1][i].GetValue();
            }
            return result;
        }
        public Neuron this[uint i, uint j]
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

        //TODO: WRITE LINE 
        public override string ToString()
        {
            StringBuilder sB = new StringBuilder();
           for(int i = 0; i < NeuronsNetwork.Length; i++)
            {
                for(int j = 0; j< NeuronsNetwork[i].Length; j++)
                {
                    sB.AppendLine(NeuronsNetwork[i][j].ToString());
                }
                sB.AppendLine();
            }
            return sB.ToString();
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network(2, 3);
            Console.WriteLine(network.ToString());
            network[0, 0].AddNextNeuron(network[1, 0],1);
            network[0, 1].AddNextNeuron(network[1, 0], 1);

            network[0, 0].AddNextNeuron(network[1, 1], 1);
            network[0, 1].AddNextNeuron(network[1, 1], 1);

            network[0, 0].AddNextNeuron(network[1, 2], 1);

            foreach (double res in network.GetResult(3, 2))
            {
                Console.WriteLine(res);
            }
            Console.ReadLine();
        }
    }
}
