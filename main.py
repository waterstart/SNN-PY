import torch
from DataTools import WaveDataReader
from NeuralNetwork import AutoEncoder
from NeuralNetwork import TreeModel
from NeuralNetwork import Model

Layers,AnswerSet = WaveDataReader()
model = AutoEncoder(NumNodes=448, LatentDimensions=32)
treemodel = TreeModel()
OtherModel = Model(NumNodes=112)
input_tensor = torch.randn(10, 488)
Layers[0] = model(Layers[0].float())
print(Layers[0].shape)
OtherModel.ReadAndTrain(Layers,AnswerSet)
treemodel.ReadAndTrain(Layers,AnswerSet)
#.venv\Scripts\Activate.ps1