using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace SpamDetection
{
    public class SpamData
    {
        [LoadColumn(0)]
        public string SpamText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Spam;
    }

    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // [ColumnName("Probability")]
        public float Probability { get; set; }

        //  [ColumnName("Score")]
        public float Score { get; set; }
    }
}
