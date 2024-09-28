using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using UglyToad.PdfPig.Graphics;

namespace LocalChatBot
{
    public class PdfData
    {
        public string Text { get; set; }
    }

    public class ModelInput
    {
        public string Text { get; set; }
        //public string Label { get; set; } // If you're doing supervised learning
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
    }

    public class MLModel
    {
        private static MLContext mlContext = new MLContext();

        public static ITransformer TrainModel(List<ModelInput> modelInputList)
        {
            // Load data into ML.NET
            IDataView dataView = mlContext.Data.LoadFromEnumerable(modelInputList);

            // Build the pipeline for text processing and classification
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text))
                            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelInput.Text))) // For classification
                            .Append(mlContext.Transforms.Concatenate("Features", "Features"))
                            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            /*var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text))
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelInput.Text)))
                                    .Append(mlContext.Transforms.Concatenate("Features", "Text"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"))
                                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                    .Append(mlContext.Transforms.CopyColumns("Features", "Label"));*/

            // Train the model
            var model = pipeline.Fit(dataView);

            return model;
        }

        public static ModelOutput Predict(ITransformer model, string input)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var result = predictionEngine.Predict(new ModelInput { Text = input });

            return new ModelOutput { Prediction = result.Prediction };
        }
    }
}