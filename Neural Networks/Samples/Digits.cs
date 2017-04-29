using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks.Samples
{

    class Digits
    {
        static NetworkREP network = new NetworkREP(Network.LearningType.Reverse_error_propagation, new Network.Functions
               (delegate (double value) { return 1 / (1 + Math.Exp(-value)); }, null, null), 63, 6, 1);
        static void CreateImages()
        {
            Random rand = new Random();
            for (int i = 0;i < COUNT; i++)
            {
                Bitmap bmp = new Bitmap("sample0.png", true);

                for (int j = 0; j< rand.Next(1, 9); j++)
                {

                    //bmp.SetPixel(rand.Next(0,6), rand.Next(0, 8), Color.Black);
                    int x = rand.Next(1, 6);
                    int y = rand.Next(1, 8);
                    bmp.SetPixel(x, y, Color.White);
                }

                bmp.Save($"Images\\sample{i}.png",System.Drawing.Imaging.ImageFormat.Png);
                bmp.Dispose();
            }

            for (int i = COUNT; i < COUNT + BAD_COUNT; i++)
            {
                Bitmap bmp = new Bitmap("sample0.png", false);

                for (int j = 0; j < 35; j++)
                {
                    int x = rand.Next(1, 6);
                    int y = rand.Next(1, 8);
                    bmp.SetPixel(x, y, Color.White);
                }

                bmp.Save($"Images\\sample{i}.png", System.Drawing.Imaging.ImageFormat.Png);
                bmp.Dispose();
            }
        }
        public static void Result()
        {
            while (true)
            {
                string filename = Console.ReadLine();
                try
                {
                    Console.WriteLine(network.GetResult(InputImage(new Bitmap(filename)))[0]);
                }
                catch
                {

                }
            }
        }

        static double[] InputImage(Bitmap bitmap)
        {
            double[] args = new double[bitmap.Width * bitmap.Height];
            int count = 0;

            for(int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    args[count] = bitmap.GetPixel(i, j).ToArgb() != Color.White.ToArgb() ? 1 : 0;
                        ++count;
                }
            }
            return args;
        }

        static int COUNT = 2500;
        static int BAD_COUNT = COUNT;

        public static void Main()
        {
            CreateImages();
            network.LearningNorm = 0.9;
            network.InertialTerm = 1;

            for(int i = 0; i < network.LayersCount - 1; i++)
            {
                for(int j = 0; j < network[i].Length; j++)
                {
                    for(int k = 0; k< network[i+1].Length; k++)
                    {
                        network[i, j].AddNextNeuron(network[i + 1, k],-0.23);
                    }
                }
            }

            int start = Environment.TickCount;
            for (int i = 0; i < COUNT; i++)
            {
                network.Learning(InputImage(new Bitmap($"Images\\sample{i}.png")), new double[] { 1 });
            }
            Console.WriteLine("good" + (Environment.TickCount - start));
            start = Environment.TickCount;
            for (int i = COUNT; i < COUNT + BAD_COUNT; i++)
            {
                network.Learning(InputImage(new Bitmap($"Images\\sample{i}.png")), new double[] { 0 });
            }
            Console.WriteLine("bad" + (Environment.TickCount - start));
            Result();

        }

    }
}
