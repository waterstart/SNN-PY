import torch
import torch.nn as NeuralNetwork
import torch.nn.functional as Functional
from sklearn.model_selection import train_test_split
import torch.optim.adam 
import matplotlib.pyplot as plt
import numpy as np
class AutoEncoder(NeuralNetwork.Module):
    def __init__(self,NumNodes, LatentDimensions = 32):
        super().__init__()
        Layers = []


        while NumNodes > 128:
            NumNodes_Half = NumNodes // 2
            if(NumNodes_Half % 2 != 0):
                NumNodes_Half-=1
            Layers.append(NeuralNetwork.Linear(NumNodes,NumNodes_Half))
            Layers.append(NeuralNetwork.ReLU())
            NumNodes=NumNodes_Half
        
        self.Encoder = NeuralNetwork.Sequential(*Layers)

    def forward(self,InputData):
            return self.Encoder(InputData)




            

       


class Model(NeuralNetwork.Module):
    def __init__(self, NumNodes, HidLayer1 = 64,HidLayer2 = 32,HidLayer3 = 32, OutLayer=5):
        super().__init__()
        self.BridgeNto1 = NeuralNetwork.Linear(NumNodes,HidLayer1)
        self.BridgeH1toH2 = NeuralNetwork.Linear(HidLayer1,HidLayer2)
        self.BridgeH2toH3 = NeuralNetwork.Linear(HidLayer2,HidLayer3)
        self.BridgeH2toOut = NeuralNetwork.Linear(HidLayer3,OutLayer)
    
    def forward(self,Node):
        Node = Functional.relu(self.BridgeNto1(Node)) 
        Node = Functional.relu(self.BridgeH1toH2(Node))
        Node = Functional.relu(self.BridgeH2toH3(Node))
        Node = self.BridgeH2toOut(Node)

        return Node
    
    def ReadAndTrain(self, FeatureSet,AnswerSet):
           ModelCriterion = NeuralNetwork.MSELoss()
           ModelOptimiser = torch.optim.Adam(self.parameters(), lr=0.001 ,weight_decay=1e-4)
           EpochAmount = 100000
           Losses = []
   


           for FeatureList, AnswerList in zip(FeatureSet, AnswerSet):

            FeatureTrain, FeatureTest, AnswerTrain, AnswerTest = train_test_split(FeatureList.detach().cpu().numpy(),
    AnswerList.detach().cpu().numpy(),test_size=0.2)
           

            FeatureTrain =  torch.FloatTensor(FeatureTrain)
            FeatureTest = torch.FloatTensor(FeatureTest)

            AnswerTrain = torch.FloatTensor(AnswerTrain)
            AnswerTest = torch.FloatTensor(AnswerTest)

            for index in range(EpochAmount):
                AnswerPredict = self.forward(FeatureTrain)
                Loss = ModelCriterion(AnswerPredict,AnswerTrain )

                Losses.append(Loss.detach().numpy())

                if index % 10000 == 0:
                        print(f'Epoch: {index} and loss: {Loss} and sqrt loss {np.sqrt(Loss.detach().numpy())}')
                
                ModelOptimiser.zero_grad()
                Loss.backward()
                ModelOptimiser.step()

            plt.plot(range(EpochAmount), Losses)
            plt.ylabel("loss/error")
            plt.xlabel('Epoch')
            
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
class TreeModel():
     def ReadAndTrain(self,FeatureSet,AnswerSet):
       
   


           for FeatureList, AnswerList in zip(FeatureSet, AnswerSet):

            FeatureTrain, FeatureTest, AnswerTrain, AnswerTest = train_test_split( FeatureList.detach().cpu().numpy(),
            AnswerList.detach().cpu().numpy(),random_state=17,test_size=0.2)
            print("feature list: ",type(FeatureList))
            print("Answer list: ", type(AnswerList))

            RFC =RandomForestRegressor()
            RFC.fit(FeatureTrain,AnswerTrain)
            AnswerPred = RFC.predict(FeatureTest)
            RFC.score(FeatureTest,AnswerTest)

            MeanScoreError = mean_squared_error(AnswerTest, AnswerPred)
            SQRMeanScoreError = np.sqrt(mean_squared_error(AnswerTest, AnswerPred))

            print("error squared bro please: ",MeanScoreError)
            print("error reqsquared bro please: ",SQRMeanScoreError)

            

         

            
            
            
           
        
        
            
        
            