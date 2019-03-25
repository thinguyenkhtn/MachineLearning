using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SpamDetection
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "SMSSpamData.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TrainCatalogBase.TrainTestData splitDataView = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseLoadedModelWithBatchItems(mlContext);
            Console.ReadKey();
        }

        public static void UseLoadedModelWithBatchItems(MLContext mlContext)
        {
            IEnumerable<SpamData> spams = new[]
            {
                new SpamData
                {
                    SpamText = "start xxx ???www.abc.tv"
                },
                new SpamData
                {
                    SpamText = "I love you."
                }
            };

            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            IDataView spamStreamingDataView = mlContext.Data.LoadFromEnumerable(spams);

            IDataView predictions = loadedModel.Transform(spamStreamingDataView);

            // Use model to predict whether spam data is Spame (1) or Ham (0).
            IEnumerable<SpamPrediction> predictedResults = mlContext.Data.CreateEnumerable<SpamPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

            IEnumerable<(SpamData spam, SpamPrediction prediction)> spamsAndPredictions = spams.Zip(predictedResults, (spam, prediction) => (spam, prediction));
            foreach ((SpamData spam, SpamPrediction prediction) item in spamsAndPredictions)
            {
                Console.WriteLine($"Spam: {item.spam.SpamText} | Prediction: {(Convert.ToBoolean(item.prediction.Prediction) ? "Spam" : "Ham")} | Probability: {item.prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SpamData.SpamText))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SpamData, SpamPrediction> predictionFunction = model.CreatePredictionEngine<SpamData, SpamPrediction>(mlContext);
            SpamData sampleStatement = new SpamData()
            {
                SpamText = "hahahahaha 08702840625.COMUK. dfdf-fdfdf dfdfdfdfdfdfdf"
            };
            var resultprediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Spam: {sampleStatement.SpamText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Spam" : "Ham")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            SaveModelAsFile(mlContext, model);
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        public static TrainCatalogBase.TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SpamData>(_dataPath, hasHeader: false);
            TrainCatalogBase.TrainTestData splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
    }
}
