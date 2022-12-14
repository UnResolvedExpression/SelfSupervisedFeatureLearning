from torch.utils.data.dataloader import DataLoader
import configs
import torch
from torch import nn
import os
from dataset import Dataset
from encDec import encDec
from encDecMask import encDecMask
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import tensorforce.agents.dqn as dqnAgent

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        #return dict(type='int', num_values=4)
        return dict(type='int', shape=(350,350)) #as large as the image - window size

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        #return super().max_episode_timesteps()
        return 1 #only 1 step for this guy as per paper

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        #Q=Q+actions*(loss-Q)
        #next_state = np.random.random(size=(8,))
        next_state=actions
        terminal=True
        #terminal = False  # Always False if no "natural" terminal state
        #reward = np.random.random()
        reward=loss
        return next_state, terminal, reward


globalLoss=0
def rewardLossFcn(img, comparison):

    loss = -torch.mean((comparison-img)**2)
    print(loss)
    return loss

def criterionFcn(img, comparison):

    loss = torch.mean((comparison-img)**2)
    globalLoss=loss
    print('criterionFcn')
    print(loss)
    return loss

def punishLossFcn(img, comparison):

    print('globalLoss')
    print(globalLoss)
    loss = torch.mean((comparison-img)**2)*10^-8+globalLoss.np()
    print('punishLossFcn')
    print(loss)
    return loss

if __name__ == '__main__':
    print("Begin deepq")
    #config = dict(device='GPU') this switches tensorforce to gpu
    CUDA_LAUNCH_BLOCKING = 1
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(configs.resultPath):
        os.makedirs(configs.resultPath)
    torch.manual_seed(configs.seed)
    studentModel = encDec().cuda()
    studentModel = studentModel.to(configs.device)
    trainDataset = Dataset(configs.dataPath)
    validationDataset = Dataset(configs.validationDataPath)
    teacherModel = encDecMask().cuda()
    teacherModel = teacherModel.to(configs.device)

    #print(trainDataset)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validationDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    studentOptimizer = torch.optim.Adam(studentModel.parameters(), lr=configs.learning_rate, weight_decay=1e-5)
    teacherOptimizer = torch.optim.Adam(teacherModel.parameters(), lr=configs.learning_rate, weight_decay=1e-5)
    counter=0
    # adapt these
    for epoch in range(configs.num_epochs):
        with tqdm(trainDataLoader) as tepoch:
            saveImg=0;
            debugImg=0;
            total_loss = 0
            totalTeacherLoss=0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                img=Variable(data).cuda()
                tempimg=teacherModel(img)


                print(tempimg.shape)
                teacherImg=torch.sum(tempimg,1)
                print(teacherImg)
                maskargmax=torch.full((2,400,400),0).cuda()
                maskargmax[1,:,:]=torch.reshape(teacherImg,(400,400))
                mask = torch.argmax(maskargmax, dim=0) #if the temping value is > 0.5, then 1 is recorded in the mask, otherwise 0
                print(mask)

                maskedImg=(img*mask)
                maskedImg.requires_grad_(requires_grad=True)

                studentOutput = studentModel(maskedImg)
                loss = criterion(studentOutput, maskedImg)

                lossImg=teacherModel(studentOutput)
                teacherLoss=rewardLossFcn(tempimg, tempimg)

                if(counter%100==0):
                    save_image(tempimg, configs.resultPath + '/tempimg_{}.png'.format(counter))
                    save_image(maskedImg, configs.resultPath + '/maskedimg_{}.png'.format(counter))
                    save_image(studentOutput, configs.resultPath + '/output_{}.png'.format(counter))

                    print("saved")
                counter+=1

                studentOptimizer.zero_grad()
                loss.backward()
                studentOptimizer.step()

                teacherOptimizer.zero_grad()
                teacherLoss.backward()
                teacherOptimizer.step()

                total_loss += loss.data
                totalTeacherLoss+=teacherLoss.data
                tepoch.set_postfix(data="train",loss=loss.item(), total_loss=total_loss/trainDataset.__len__(),
                                   teacherLoss=teacherLoss,totalTeacherLoss=totalTeacherLoss/trainDataset.__len__())
                saveImg=studentOutput
            save_image(saveImg, configs.resultPath + '/image_train_{}.png'.format(epoch))
