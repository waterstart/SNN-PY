import torch
import torch.nn as NeuralNetwork
import torch.nn.functional as Functional

class Model(NeuralNetwork.Module):
    def __init__(self, NumNodes, HidLayer1 = 8,HidLayer2 = 9, OutLayer=5):
        super().__init__()
        self.BridgeNto1 = NeuralNetwork.Linear(NumNodes,HidLayer1)
        self.BridgeH1toH2 = NeuralNetwork.Linear(HidLayer1,HidLayer2)
        self.BridgeH2toOut = NeuralNetwork.Linear(HidLayer2,OutLayer)
    
    def Forward(self,Node):
        Node = Functional.relu(self.BridgeNto1)
        Node = Functional.relu(self.BridgeH1toH2)
        Node = self.BridgeH2toOut(Node)

        return Node