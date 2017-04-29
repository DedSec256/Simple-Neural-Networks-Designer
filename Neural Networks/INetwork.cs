using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Networks
{
    interface INetwork
    {
        void Learning(double[] args, double[] expectedRes);
    }
}
