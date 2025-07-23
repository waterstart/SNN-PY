import torch
from DataTools import WaveDataReader
from NeuralNetwork import Model

TestModel = Model(448)
Model.ReadAndTrain(TestModel,*WaveDataReader())
#.venv\Scripts\Activate.ps1