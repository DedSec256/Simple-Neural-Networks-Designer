using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks
{
    public static class Outputter
    {
        public static void Log(string text)
        {
            Console.WriteLine(text);
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
}
