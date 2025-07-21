import torch
import os
from pathlib import Path
import numpy
import pandas as pd

def WaveDataReader():
    SpectralPath = os.getcwd() + "/Spectral training data/Parsed"
    os.chdir(SpectralPath)
    TensorArray = []
    for files in Path.cwd().iterdir():
        with open(files.name,"r") as file:
            TrainingData = pd.read_csv(file)
            TensorTrainingData = torch.tensor(TrainingData.to_numpy(), dtype=torch.float64)
            TensorArray.append(TensorTrainingData)
            print(TensorTrainingData)
    print(SpectralPath)

WaveDataReader()