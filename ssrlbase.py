from torch.utils.data.dataloader import DataLoader
import configs
import torch
from torch import nn
import os
from dataset import Dataset
from encDec import encDec
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import random


if __name__ == '__main__':
    print("Begin ssrlbase")
    if not os.path.exists(configs.resultPath):
        os.makedirs(configs.resultPath)
    torch.manual_seed(configs.seed)
    model = encDec().cuda()
    model = model.to(configs.device)
    trainDataset = Dataset(configs.dataPath)
    validationDataset = Dataset(configs.validationDataPath)

    #print(trainDataset)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validationDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=1e-5)

    # adapt these
    for epoch in range(configs.num_epochs):
        with tqdm(trainDataLoader) as tepoch:
            saveImg=0;
            debugImg=0;
            total_loss = 0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                #data, target = data.to(configs.device), target.to(configs.device)
                #optimizer.zero_grad()
                #output = model(data)
                #print(trainDataLoader.dataset)
                #for data in trainDataLoader: #why is this nested inside...
                img = data
                rx=random.randrange(0,300)
                ry=random.randrange(0,300)
                cutout= img[:,:,rx:rx+100,ry:ry+100].clone()
                maskedImg=img.clone()
                maskedImg[:,:,rx:rx+100,ry:ry+100]=torch.zeros(1,3,100, 100)
                #save_image(cutout, configs.resultPath + '/cutout.png')
                #save_image(img, configs.resultPath + '/imgmasked.png')

                maskedImg = Variable(maskedImg).cuda()
                img=Variable(img).cuda()
                #saveImg=img;
                #print("img")
                #print(img)
                #print(img.size())
                # ===================forward=====================
                output = model(maskedImg)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                tepoch.set_postfix(data="train",loss=loss.item(), total_loss=total_loss/trainDataset.__len__())
                saveImg=output
            save_image(saveImg, configs.resultPath + '/image_train_{}.png'.format(epoch))

                #now add in some validation
                #then ssrl
                    #square task
                    #noise task
        torch.cuda.empty_cache()
        valLoss=0
        with tqdm(validationDataLoader) as tepoch:
            saveImg=0;
            total_loss = 0
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                #data, target = data.to(configs.device), target.to(configs.device)
                #optimizer.zero_grad()
                #output = model(data)
                #print(trainDataLoader.dataset)
                #for data in trainDataLoader: #why is this nested inside...
                img = data
                rx = random.randrange(0, 300)
                ry = random.randrange(0, 300)
                cutout = img[:, :, rx:rx + 100, ry:ry + 100].clone()
                maskedImg = img.clone()
                maskedImg[:, :, rx:rx + 100, ry:ry + 100] = torch.zeros(1, 3, 100, 100)
                # save_image(cutout, configs.resultPath + '/cutout.png')
                # save_image(img, configs.resultPath + '/imgmasked.png')

                maskedImg = Variable(maskedImg).cuda()
                img = Variable(img).cuda()
                # saveImg=img;
                # print("img")
                # print(img)
                # print(img.size())
                # ===================forward=====================
                output = model(maskedImg)
                loss = criterion(output, img)
                # ===================backward====================
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                total_loss += loss.data
                tepoch.set_postfix(data="val",loss=loss.item(), total_loss=total_loss/validationDataset.__len__())
                #output = Variable(output).cuda()
                saveImg=output
                #save_image(output, configs.resultPath + '/image_{}.png'.format(epoch))


            save_image(saveImg, configs.resultPath+'/image_val_{}.png'.format(epoch))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), configs.resultPath+'/conv_autoencoder_{}_{}.pth'.format(epoch,valLoss))

            #sleep(0.1)
                #total_loss = 0
            #print(trainDataLoader.dataset)
            #for data in trainDataLoader:
                #img = data
                #img = Variable(img).cuda()
                #print("img")
                #print(img)
                #print(img.size())
                # ===================forward=====================

                #output = model(img)
                #loss = criterion(output, img)
                # ===================backward====================
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                #total_loss += loss.data
            # ===================log========================
            # print('epoch [{}/{}], loss:{:.4f}'
            #       .format(epoch + 1, configs.num_epochs, total_loss))



    # # adapt these
    # for epoch in range(configs.num_epochs):
    #     total_loss = 0
    #     #print(trainDataLoader.dataset)
    #     for data in trainDataLoader:
    #         img = data
    #         img = Variable(img).cuda()
    #         #print("img")
    #         #print(img)
    #         #print(img.size())
    #         # ===================forward=====================
    #
    #         output = model(img)
    #         loss = criterion(output, img)
    #         # ===================backward====================
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.data
    #     # ===================log========================
    #     print('epoch [{}/{}], loss:{:.4f}'
    #           .format(epoch + 1, configs.num_epochs, total_loss))
    #     if epoch % 10 == 0:
    #         # pic = to_img(output.cpu().data)#why is this cpu, because it is an external function that scales the img
    #         save_image(img, configs.resultPath+'/image_{}.png'.format(epoch))
    #
    # torch.save(model.state_dict(), './conv_autoencoder.pth')