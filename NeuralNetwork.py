import torch
import torch.nn as NeuralNetwork
import torch.nn.functional as Functional
from sklearn.model_selection import train_test_split
import torch.optim.adam 
class Model(NeuralNetwork.Module):
    def __init__(self, NumNodes, HidLayer1 = 64,HidLayer2 = 32, OutLayer=5):
        super().__init__()
        self.BridgeNto1 = NeuralNetwork.Linear(NumNodes,HidLayer1)
        self.BridgeH1toH2 = NeuralNetwork.Linear(HidLayer1,HidLayer2)
        self.BridgeH2toOut = NeuralNetwork.Linear(HidLayer2,OutLayer)
    
    def Forward(self,Node):
        Node = Functional.relu(self.BridgeNto1(Node)) 
        Node = Functional.relu(self.BridgeH1toH2(Node))
        Node = self.BridgeH2toOut(Node)

        return Node
    
    def ReadAndTrain(self, FeatureSet,AnswerSet):
           ModelCriterion = NeuralNetwork.MSELoss()
           ModelOptimiser = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
           EpochAmount = 10000000000000
           Losses = []
   


           for FeatureList, AnswerList in zip(FeatureSet, AnswerSet):

            FeatureTest,FeatureTrain,AnswerTest,AnswerTrain = train_test_split(FeatureList,AnswerList,test_size=0.2)
           

            FeatureTrain =  torch.FloatTensor(FeatureTrain)
            FeatureTest = torch.FloatTensor(FeatureTest)

            AnswerTrain = torch.FloatTensor(AnswerTrain)
            AnswerTest = torch.FloatTensor(AnswerTest)

            for index in range(EpochAmount):
                AnswerPredict = torch.FloatTensor(self.Forward(FeatureTrain))
                Loss = ModelCriterion(AnswerPredict,AnswerTrain )

                Losses.append(Loss.detach().numpy())

                if index % 10000 == 0:
                        print(f'Epoch: {index} and loss: {Loss}')
                
                ModelOptimiser.zero_grad()
                Loss.backward()
                ModelOptimiser.step()

            

            
            
           
        
        
            
        
            