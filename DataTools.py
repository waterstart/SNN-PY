import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import numpy
import pandas as pd

def WaveDataReader():
    SpectralPath = os.getcwd() + "/Spectral training data/Parsed"
    os.chdir(SpectralPath)
    FeatureSet = []
    AnswerSet = []
    for files in Path.cwd().iterdir():
        with open(files.name,"r") as file:
            TrainingData = pd.read_csv(file, header=None)
            for index, Feature in enumerate(TrainingData.iloc[0]):
                 try:
                     float(Feature)
                 except ValueError:
                      TrainingDataFeatures = TrainingData.iloc[1:,:index].to_numpy()

                      FeatureSet.append(TrainingDataFeatures)
                      AnswerDataFeatures = (TrainingData.iloc[1:, index:].apply(pd.to_numeric, errors='coerce')).to_numpy()

                      AnswerSet.append(AnswerDataFeatures)
                      break                
    return FeatureSet,  AnswerSet
