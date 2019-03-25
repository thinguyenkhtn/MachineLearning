using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace SpamDetection
{
    class SpamInput
    {
        [LoadColumn(0)]
        public string Label { get; set; }
        [LoadColumn(1)]
        public string Message { get; set; }
    }
}
