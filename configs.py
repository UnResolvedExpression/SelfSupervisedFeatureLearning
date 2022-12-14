import torch
dataPath="C:/Users/ghait/Repos/SSFLData/train"
validationDataPath="C:/Users/ghait/Repos/SSFLData/val"
resultPath="C:/Users/ghait/Repos/SSFLData/results2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate=0.0005
num_epochs=1500
seed=1
batch_size=1
threads=1