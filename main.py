import torch
from DataTools import WaveDataReader
from RandomForest import TreeModel

TestModel = TreeModel()
TestModel.ReadAndTrain(*WaveDataReader())
#.venv\Scripts\Activate.ps1