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

#the student output can then be put back into the teacher and then we can compare the first output of the teacher to the second one
#we can add another loss before that which lightly punishes the teacher for how much masking they do
globalLoss=0
def rewardLossFcn(img, comparison):
    # specifying the batch size
    #my_batch_size = my_outputs.size()[0]
    # calculating the log of softmax values
    #my_outputs = F.log_softmax(my_outputs, dim=1)
    # selecting the values that correspond to labels
    #my_outputs = my_outputs[range(my_batch_size), my_labels]
    # returning the results
    # number_examples=1
    # return -torch.sum(my_outputs) / number_examples
    #loss=criterion(img,comparison)
    loss = -torch.mean((comparison-img)**2)
    print(loss)
    return loss

def criterionFcn(img, comparison):
    # specifying the batch size
    #my_batch_size = my_outputs.size()[0]
    # calculating the log of softmax values
    #my_outputs = F.log_softmax(my_outputs, dim=1)
    # selecting the values that correspond to labels
    #my_outputs = my_outputs[range(my_batch_size), my_labels]
    # returning the results
    # number_examples=1
    # return -torch.sum(my_outputs) / number_examples
    #loss=criterion(img,comparison)
    loss = torch.mean((comparison-img)**2)
    globalLoss=loss
    print('criterionFcn')
    print(loss)
    return loss

def punishLossFcn(img, comparison):
    # specifying the batch size
    #my_batch_size = my_outputs.size()[0]
    # calculating the log of softmax values
    #my_outputs = F.log_softmax(my_outputs, dim=1)
    # selecting the values that correspond to labels
    #my_outputs = my_outputs[range(my_batch_size), my_labels]
    # returning the results
    # number_examples=1
    # return -torch.sum(my_outputs) / number_examples
    #loss=criterion(img,comparison)
    #loss = + 0.0000000001
    print('globalLoss')
    print(globalLoss)
    loss = torch.mean((comparison-img)**2)*10^-8+globalLoss.np()
    print('punishLossFcn')
    print(loss)
    return loss

if __name__ == '__main__':
    print("Begin ssfl")
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
                #data, target = data.to(configs.device), target.to(configs.device)
                #optimizer.zero_grad()
                #output = model(data)
                #print(trainDataLoader.dataset)
                #for data in trainDataLoader: #why is this nested inside...
                #img = data
                #rx=random.randrange(0,300)
                #ry=random.randrange(0,300)
                #cutout= img[:,:,rx:rx+100,ry:ry+100].clone()
                #maskedImg=img.clone()
                #maskedImg[:,:,rx:rx+100,ry:ry+100]=torch.zeros(1,3,100, 100)
                #save_image(cutout, configs.resultPath + '/cutout.png')
                #save_image(img, configs.resultPath + '/imgmasked.png')
                # ===================masking=====================
                img=Variable(data).cuda()
                tempimg=teacherModel(img)
                # punishmentLoss=punishLossFcn(tempimg,img)
                # torch.autograd.set_detect_anomaly(True)
                # teacherOptimizer.zero_grad()
                # punishmentLoss.backward()
                # teacherOptimizer.step()


                print(tempimg.shape)
                teacherImg=torch.sum(tempimg,1)
                print(teacherImg)
                maskargmax=torch.full((2,400,400),0).cuda()
                maskargmax[1,:,:]=torch.reshape(teacherImg,(400,400))
                mask = torch.argmax(maskargmax, dim=0) #if the temping value is > 0.5, then 1 is recorded in the mask, otherwise 0
                #print(maskargmax.device)
                print(mask)
                #print(mask.device)
                # save_image(mask.type(torch.FloatTensor), configs.resultPath + '/masked_{}.png'.format(counter))
                #mask = Variable(maskedImg).cuda()
                maskedImg=(img*mask)
                maskedImg.requires_grad_(requires_grad=True)
                #save_image(maskedImg.type(torch.FloatTensor), configs.resultPath + '/maskedImg_{}.png'.format(counter))

                #saveImg=img;
                #print("img")
                #print(img)
                #print(img.size())
                # ===================forward=====================
                studentOutput = studentModel(maskedImg)
                #output.requires_grad_(requires_grad=True)
                #save_image(output.type(torch.FloatTensor), configs.resultPath + '/outputImg_{}.png'.format(counter))
                loss = criterion(studentOutput, maskedImg)
                #teacherLoss=loss.clone()
                #teacherLoss= criterion(img,img)
                #print(loss)
                #teacherLoss.data=torch.tensor(loss.item()**-1)
                #print(teacherLoss)

                #trying out softmax
                lossImg=teacherModel(studentOutput)
                teacherLoss=rewardLossFcn(tempimg, tempimg)
                    #criterion(output, img)
                    #torch.tensor(0.01)
                    #criterion(tempimg,img)
                if(counter%100==0):
                    save_image(tempimg, configs.resultPath + '/tempimg_{}.png'.format(counter))
                    save_image(maskedImg, configs.resultPath + '/maskedimg_{}.png'.format(counter))
                    save_image(studentOutput, configs.resultPath + '/output_{}.png'.format(counter))

                    print("saved")
                counter+=1




                # ===================teacher antagonistic loss=====================
                # torch.autograd.set_detect_anomaly(True)
                # teacherOptimizer.zero_grad()
                # teacherLoss.backward()
                # teacherOptimizer.step()
                #teacher will make a mask
                #the loss will be teacherloss=criterion(outputfromstudentwitheverything but area of interest masked,inputwebaoim)
                #criterion will be the opposite of mse


                # ===================backward====================
                studentOptimizer.zero_grad()
                loss.backward()
                studentOptimizer.step()

                teacherOptimizer.zero_grad()
                teacherLoss.backward()
                teacherOptimizer.step()
                            #in the first place, the teacher fxn was not configured to place a mask

                total_loss += loss.data
                totalTeacherLoss+=teacherLoss.data
                tepoch.set_postfix(data="train",loss=loss.item(), total_loss=total_loss/trainDataset.__len__(),
                                   teacherLoss=teacherLoss,totalTeacherLoss=totalTeacherLoss/trainDataset.__len__())
                saveImg=studentOutput
            save_image(saveImg, configs.resultPath + '/image_train_{}.png'.format(epoch))



                #now add in some validation
                #then ssrl
                    #square task
                    #noise task
    #     torch.cuda.empty_cache()
    #     valLoss=0
    #     with tqdm(validationDataLoader) as tepoch:
    #         saveImg=0;
    #         total_loss = 0
    #         for data in tepoch:
    #             tepoch.set_description(f"Epoch {epoch}")
    #             #data, target = data.to(configs.device), target.to(configs.device)
    #             #optimizer.zero_grad()
    #             #output = model(data)
    #             #print(trainDataLoader.dataset)
    #             #for data in trainDataLoader: #why is this nested inside...
    #             img = data
    #             rx = random.randrange(0, 300)
    #             ry = random.randrange(0, 300)
    #             cutout = img[:, :, rx:rx + 100, ry:ry + 100].clone()
    #             maskedImg = img.clone()
    #             maskedImg[:, :, rx:rx + 100, ry:ry + 100] = torch.zeros(1, 3, 100, 100)
    #             # save_image(cutout, configs.resultPath + '/cutout.png')
    #             # save_image(img, configs.resultPath + '/imgmasked.png')
    #
    #             maskedImg = Variable(maskedImg).cuda()
    #             img = Variable(img).cuda()
    #             # saveImg=img;
    #             # print("img")
    #             # print(img)
    #             # print(img.size())
    #             # ===================forward=====================
    #             output = studentModel(maskedImg)
    #             loss = criterion(output, img)
    #             # ===================backward====================
    #             #optimizer.zero_grad()
    #             #loss.backward()
    #             #optimizer.step()
    #             total_loss += loss.data
    #             tepoch.set_postfix(data="val",loss=loss.item(), total_loss=total_loss/validationDataset.__len__())
    #             #output = Variable(output).cuda()
    #             saveImg=output
    #             #save_image(output, configs.resultPath + '/image_{}.png'.format(epoch))
    #
    #
    #         save_image(saveImg, configs.resultPath+'/image_val_{}.png'.format(epoch))
    #
    #     if epoch % 10 == 0:
    #         torch.save(studentModel.state_dict(), configs.resultPath + '/conv_autoencoder_{}_{}.pth'.format(epoch, valLoss))
    #
    #         #sleep(0.1)
    #             #total_loss = 0
    #         #print(trainDataLoader.dataset)
    #         #for data in trainDataLoader:
    #             #img = data
    #             #img = Variable(img).cuda()
    #             #print("img")
    #             #print(img)
    #             #print(img.size())
    #             # ===================forward=====================
    #
    #             #output = model(img)
    #             #loss = criterion(output, img)
    #             # ===================backward====================
    #             #optimizer.zero_grad()
    #             #loss.backward()
    #             #optimizer.step()
    #             #total_loss += loss.data
    #         # ===================log========================
    #         # print('epoch [{}/{}], loss:{:.4f}'
    #         #       .format(epoch + 1, configs.num_epochs, total_loss))
    #
    #
    #
    # # # adapt these
    # # for epoch in range(configs.num_epochs):
    # #     total_loss = 0
    # #     #print(trainDataLoader.dataset)
    # #     for data in trainDataLoader:
    # #         img = data
    # #         img = Variable(img).cuda()
    # #         #print("img")
    # #         #print(img)
    # #         #print(img.size())
    # #         # ===================forward=====================
    # #
    # #         output = model(img)
    # #         loss = criterion(output, img)
    # #         # ===================backward====================
    # #         optimizer.zero_grad()
    # #         loss.backward()
    # #         optimizer.step()
    # #         total_loss += loss.data
    # #     # ===================log========================
    # #     print('epoch [{}/{}], loss:{:.4f}'
    # #           .format(epoch + 1, configs.num_epochs, total_loss))
    # #     if epoch % 10 == 0:
    # #         # pic = to_img(output.cpu().data)#why is this cpu, because it is an external function that scales the img
    # #         save_image(img, configs.resultPath+'/image_{}.png'.format(epoch))
    # #
    # # torch.save(model.state_dict(), './conv_autoencoder.pth')
