using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Keras;
using Keras.Helper;
using Keras.Layers;
using Keras.Models;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Numpy;
using NumSharp;
using Python.Runtime;
using Tensorflow;
using static Tensorflow.Binding;
using np = Numpy.np;
using Shape = Keras.Shape;

namespace SplitDeploymentTesting {
    public class Program {
        public static void Main(string[] args) {
            // CreateHostBuilder(args).Build().Run();
            //CreateWebHostBuilder(args).Build().Run();
            Sequential Seq = new Sequential();
            Seq.Add(new Dense(32, activation: "relu", input_shape: new Shape(250,250,3)));
            Seq.Add(new Dense(64, activation: "relu"));
            Seq.Add(new Dense(1, activation: "sigmoid"));
            
            Console.WriteLine(Backend.GetBackend());
            var function = Backend.Function(Seq.Layers(0), Seq.Layers(1));
            Console.WriteLine(function);
            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });
            var val = (42, 2);
            // var data = tf.ones(new TensorShape(new int[] {1, 259, 259, 3})); 
            
            var data = Backend.ones();
            KerasIterator iter = new KerasIterator(data);
            var z = new PyIter(iter.PyObject);
            z.MoveNext();
            var output = z.Current;
            var res = function(data);
            Console.WriteLine("function Results:");
            Console.WriteLine(res);

        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder => { webBuilder.UseStartup<Startup>(); });
    }
}