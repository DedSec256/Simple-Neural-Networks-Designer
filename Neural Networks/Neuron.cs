using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Neural_Networks
{
    public abstract class Neuron
    {
        public enum Types
        {
            Input, Hidden, Output
        }
        public Types Type
        {
            get; private set;
        }

        public bool WriteLog;

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
        protected double ErrorSummary;
        protected int  ErrorsLeft;
        public double GetValue() { return Value; }
        public virtual void AsOffset()
        {
            throw new InvalidOperationException($"Нейрон {NeuronName} нельзя сделать смещением.");
        }

        public virtual void Clear()
        {
            Value = ErrorSummary =  0;
            AdmissionsLeft = MaxAdmissionsLeft;
            ErrorsLeft     = NextNeurons.Count;
        }

        public virtual void AddNextNeuron(Neuron neuron, double? weight = null)
        {
            if (NextNeurons.ContainsKey(neuron))
            {
                if (WriteLog)
                    Outputter.Warning($"Нейроны {NeuronName} и {neuron.NeuronName} уже обьединены прямой связью.");
            }

            else
            {
                Random rand = new Random();
                weight = weight == null ? rand.NextDouble() : weight;
                NextNeurons.Add(neuron, weight.Value);
                neuron.AddLastNeuron(this, weight.Value);
                neuron.AdmissionsLeft = ++neuron.MaxAdmissionsLeft;
            }
        }
        public virtual void RemoveNextNeuron(Neuron neuron)
        {
            NextNeurons.Remove(neuron);
        }
        public virtual void AddLastNeuron(Neuron neuron, double weight)
        {
            if (LastNeurons.ContainsKey(neuron))
            {
                if (WriteLog)
                    Outputter.Warning($"Нейроны {NeuronName} и {neuron.NeuronName} уже обьединены обратной связью.");
            }
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
            }
        }
        public virtual void RecieveErrorSigmoid(Neuron errorNeuron,
            double learningNorm, double inertialTerm, double? expectedValue = null)
        {
            Interlocked.Exchange(ref ErrorSummary, ErrorSummary + errorNeuron.ErrorSummary);
            Interlocked.Decrement(ref ErrorsLeft);

            if (ErrorsLeft == 0 )
            {
                ErrorSummary = Value * (1 - Value) * ErrorSummary * NextNeurons[errorNeuron];

                if (WriteLog)
                {
                    Outputter.Log($"Вес {NeuronName} - {errorNeuron.NeuronName} изменён {NextNeurons[errorNeuron]} - ");
                }

                errorNeuron.LastNeurons[this] = NextNeurons[errorNeuron] = NextNeurons[errorNeuron] * inertialTerm
                    + learningNorm * Value * errorNeuron.ErrorSummary;

                if (WriteLog)
                {
                    Outputter.Log($"{NextNeurons[errorNeuron]}");
                }

                if (LastNeurons.Count != 0)
                {
                    for (int i = 0; i < LastNeurons.Keys.Count; i++)
                    {
                        LastNeurons.Keys.ElementAt(i).RecieveErrorSigmoid(this, learningNorm, inertialTerm);
                    }
                }
            }
        }

        public virtual void SendSignal(Network.ActivateFunction activFunc)
        {
            Parallel.ForEach(NextNeurons, nextNeuron =>
            {
                double res = activFunc(Value);
                nextNeuron.Key.RecieveSignal(res * nextNeuron.Value, activFunc);
                if (WriteLog)
                    Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName} отправлен = {Value} * {nextNeuron.Value} -> {res} * {nextNeuron.Value}");

                Value = res;
            }
            );

        }

        public Neuron(string neuronName, Types type)
        {
            NeuronName = neuronName;
            Type = type;

            MaxAdmissionsLeft = AdmissionsLeft = 0;
            Value = 0;

            WriteLog = false;

            NextNeurons = new Dictionary<Neuron, double>();
            LastNeurons = new Dictionary<Neuron, double>();
        }
        public override string ToString()
        {
            StringBuilder sB = new StringBuilder();
            sB.AppendLine(NeuronName);
            sB.AppendLine($"Ожидается входящих сигналов: {MaxAdmissionsLeft}");

            foreach(var obj in NextNeurons)
            {
                sB.AppendLine($"    Вес прямой связи с {obj.Key.NeuronName}: {obj.Value}");
            }

            return sB.ToString();
        }
    }
    class InputNeuron : Neuron
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
                if (WriteLog)
                {
                    Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName} отправлен = {nextNeuron.Value * Value}");
                }
            });
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

        public override void Clear()
        {
            if (!IsOffset)
                base.Clear();
            else
            {
                ErrorsLeft = NextNeurons.Count;
                return;
            }
        }
        public override void SendSignal(Network.ActivateFunction activFunc)
        {
            if (!IsOffset)
                base.SendSignal(activFunc);
            else
            {
                Parallel.ForEach(NextNeurons, nextNeuron =>
                {
                    nextNeuron.Key.RecieveSignal(nextNeuron.Value * Value, activFunc);
                    if (WriteLog)
                    {
                        Outputter.Log($"Сигнал {NeuronName} - {nextNeuron.Key.NeuronName}" +
                        $" отправлен = {nextNeuron.Value * Value}");
                    }
                }
                );
            }
        }

        public override void AsOffset()
        {
            IsOffset = true;
            NeuronName += " (OFFSET)";
            MaxAdmissionsLeft = AdmissionsLeft = 0;
            Value = 1;

            if (LastNeurons.Keys.Count != 0)
            {
                if (WriteLog)
                    Outputter.Warning($"При изменении свойства IsOffset нейрон {NeuronName} потерял связи с" +
                    $" {LastNeurons.Keys.Count} нейронами!");
            }

            while (LastNeurons.Keys.Count != 0)
            {
                LastNeurons.ElementAt(0).Key.RemoveNextNeuron(this);
                LastNeurons.Remove(LastNeurons.ElementAt(0).Key);
            }
        }
    }
    class OutputNeuron : Neuron
    {
        public uint NeuronId
        {
            get; private set;
        }
        public OutputNeuron(string neuronName, uint neuronId) : base(neuronName, Types.Input)
        {
            LastNeurons = new Dictionary<Neuron, double>();
            NeuronId = neuronId;
        }

        public override void AddNextNeuron(Neuron neuron, double? weight)
        {
            throw new Exception($"Нельзя соединить нейроны {neuron.NeuronName} и {NeuronName} прямой связью.");
        }

        public override void RecieveErrorSigmoid(Neuron errorNeuron,
            double learningNorm, double inertialTerm, double? expectedValue = null)
        {
            if (expectedValue is null) throw new ArgumentException("Значение ожидаемых выходных данных не может быть null");
            ErrorSummary = (expectedValue.Value - Value) * (Value) * (1 - Value);

                for(int i = 0; i < LastNeurons.Keys.Count; i++)
                {
                LastNeurons.Keys.ElementAt(i).RecieveErrorSigmoid(this, learningNorm, inertialTerm);
                }
        }
        public override void RecieveSignal(double value, Network.ActivateFunction activFunc)
        {
            Interlocked.Exchange(ref Value, Value + value);
            Interlocked.Decrement(ref AdmissionsLeft);
            if (AdmissionsLeft == 0) Value = activFunc(Value);
        }
    }

}


