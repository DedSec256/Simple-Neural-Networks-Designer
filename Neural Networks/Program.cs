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
            Console.WriteLine(text);
        }
    }

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
        protected int MaxAdmissionsLeft;
        protected int AdmissionsLeft;
        public int GetRemainingSignals()
        {
            return MaxAdmissionsLeft;
        }

        protected double Value;
        public double GetValue() { return Value; }
        public virtual void AsOffset() {
            throw new InvalidOperationException($"Нейрон {NeuronName} нельзя сделать смещением.");
                }

        public virtual void AddNextNeuron(Neuron neuron, double weight)
        {
            if (NextNeurons.ContainsKey(neuron)) throw new
                    InvalidOperationException($"Нейроны {NeuronName} и {neuron.NeuronName} уже обьединены прямой связью.");
            else
            {
                NextNeurons.Add(neuron, weight);
                neuron.AddLastNeuron(this, weight);
                neuron.AdmissionsLeft = ++neuron.MaxAdmissionsLeft;
            }
        }
        public virtual void RemoveNextNeuron(Neuron neuron)
        {
            NextNeurons.Remove(neuron);
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
            if (AdmissionsLeft == 0)
            {
                SendSignal(activFunc);
                Interlocked.Exchange(ref AdmissionsLeft, MaxAdmissionsLeft);
            }
        }

        private object sendSignalLock = new object();
        //TODO: --AdmissionLeft
        public virtual void SendSignal(Network.ActivateFunction activFunc)
        {
            Parallel.ForEach(NextNeurons, nextNeuron =>
            {
                double res = activFunc(Value);
                nextNeuron.Key.RecieveSignal(res * nextNeuron.Value, activFunc);
                Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName} отправлен = {Value} * {nextNeuron.Value} -> {res} * {nextNeuron.Value}");
            }
            );
            
        }
        public Neuron(string neuronName, Types type)
        {
            NeuronName = neuronName;
            Type = type;

            MaxAdmissionsLeft = AdmissionsLeft = 0;
            Value = 0;

            NextNeurons = new Dictionary<Neuron, double>();
            LastNeurons = new Dictionary<Neuron, double>();
        }
        public override string ToString()
        {
            StringBuilder sB = new StringBuilder();
            sB.AppendLine(NeuronName);
            sB.AppendLine($"Ожидается входящих сигналов: {MaxAdmissionsLeft}");

            return sB.ToString();
        }
    }

    class InputNeuron: Neuron
    {
        public uint NeuronId
        {
            get; private set;
        }

        public InputNeuron(string neuronName, uint neuronId) : base(neuronName, Types.Input)
        {
            NextNeurons = new Dictionary<Neuron, double>();
            MaxAdmissionsLeft = AdmissionsLeft = 1;
            NeuronId = neuronId;
        }
        public override void SendSignal(Network.ActivateFunction activFunc)
        {
            Parallel.ForEach(NextNeurons, nextNeuron =>
            {
                nextNeuron.Key.RecieveSignal(nextNeuron.Value * Value, activFunc);
                Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName} отправлен = {nextNeuron.Value * Value}");
            }
            );
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

        public override void SendSignal(Network.ActivateFunction activFunc)
        {
            if(!IsOffset)
                base.SendSignal(activFunc);
            else
            {
                Parallel.ForEach(NextNeurons, nextNeuron =>
                {
                    nextNeuron.Key.RecieveSignal(nextNeuron.Value * Value, activFunc);
                    Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName}" +
                        $" отправлен = {nextNeuron.Value * Value}");
                }
                );
            }
        }

        public override void AsOffset()
        {
            IsOffset = true;
            MaxAdmissionsLeft = AdmissionsLeft = 0;
            Value = 1;
            while(LastNeurons.Keys.Count != 0)
            {
                LastNeurons.ElementAt(0).Key.RemoveNextNeuron(this);
                LastNeurons.Remove(LastNeurons.ElementAt(0).Key);
            }
            NeuronName += " (OFFSET)";
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

        public override void RecieveSignal(double value, Network.ActivateFunction activFunc)
        {
            Interlocked.Exchange(ref Value, Value + value);
            Interlocked.Decrement(ref AdmissionsLeft);
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
                NeuronsNetwork[0][i] = new InputNeuron($"Входной нейрон [{0}][{i}]", i);
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
        public void TryToConfigure()
        {
            /*TODO: Проверка выходных нейронов на входящие 
             *      Проверка скрытых нейронов на входящие и исходящие
             *      проверка входных на исходящие
             * */
        }
        //TODO: Не забыть offset
        public double[] GetResult(params double[] args)
        {
            if(args.Length != NeuronsNetwork[0].Length)
            {
                throw new ArgumentException("Количество входных данных не соответсвует размеру входного слоя.");
            }

            Outputter.Log($"Входные данные: ");
            for(int i=0; i<args.Length; i++) Outputter.Log($"{NeuronsNetwork[0][i].NeuronName} = {args[i]}");
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

    class Program
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
                Network network = new Network(new Network.Functions(MyActivateFunc), 2, 4, 1, 1);

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
                    Outputter.Log("Ответ: " + res + " => " + MyActivateFunc(res));
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            Console.ReadLine();
        }
    }
}
