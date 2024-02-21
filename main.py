from dataset import WholeDataLoader
from dotted_dict import DottedDict
from torch.utils.data import DataLoader,dataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


LR = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class convnet(nn.Module):
    def __init__(self,num_classes=10):
        super(convnet,self).__init__()
        self.bn0     = nn.LayerNorm([3,28,28])
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        feat_out = x  
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7

        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0),-1)
        y_low = self.fc(feat_low)

        return feat_out, y_low

class SingleAdversery(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(SingleAdversery, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.LayerNorm([input_ch,14,14])
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        self.softmax    = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)

        return px

class Adversery(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(Adversery, self).__init__()
        #three adversery networks
        self.pred_r = SingleAdversery(input_ch, num_classes)
        self.pred2_g = SingleAdversery(input_ch, num_classes)
        self.pred3_b = SingleAdversery(input_ch, num_classes)
    def forward(self, x):
        px_r = self.pred_r(x).unsqueeze(1)
        px_g = self.pred2_g(x).unsqueeze(1)
        px_b = self.pred3_b(x).unsqueeze(1)

        return px_r, px_g, px_b

class AdverserialLoss(nn.Module):
    def __init__(self):
        super(AdverserialLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    def forward(self, px_r,px_g,px_b , label_image):
        loss = self.criterion(px_r.squeeze(), label_image[:,0,:,:].long())
        loss += self.criterion(px_g.squeeze(), label_image[:,1,:,:].long())
        loss += self.criterion(px_b.squeeze(), label_image[:,2,:,:].long())
        return loss





if __name__ == "__main__":
    
    option_train  = DottedDict({"data_dir":"data/colored_mnist", "color_var":0.020,"data_split":"train"})
    option_test  = DottedDict({"data_dir":"data/colored_mnist", "color_var":0.020,"data_split":"test"})
        
    
    colored_minist_dataset_train = WholeDataLoader(option_train)
    colored_minist_dataset_test = WholeDataLoader(option_test)

    
    colored_dataloader_train = DataLoader(colored_minist_dataset_train, num_workers= 4, persistent_workers = True,batch_size = 1024,shuffle=True)
    colored_dataloader_test = DataLoader(colored_minist_dataset_test, num_workers= 4, persistent_workers = True,batch_size = 1024,shuffle=False)
    
    model = convnet().to(device)
    discrim = Adversery().to(device)
       

    optimizer_target = torch.optim.AdamW(model.parameters(),lr = LR)
    optimizer_discriminator = torch.optim.AdamW(discrim.parameters(),lr = LR)
 

    criterion = nn.CrossEntropyLoss().to(device)
    discrim_criterion = AdverserialLoss().to(device)
    model = model.to(device)

    for epoch in range(10):
        model.train()
        accuracy,sum_loss,total_count = 0,0,0


        if epoch >= 0:
            #train the discrim network's network
            adv_loss,adv_count = 0,0
            for spurious_image,label_image,label in tqdm(colored_dataloader_train):
                
                spurious_image,label_image,label = spurious_image.to(device),label_image.to(device),label.to(device)
                optimizer_discriminator.zero_grad()
                
                optimizer_target.zero_grad()
                
                featout, logits = model(spurious_image)
                px_r,px_g,px_b = discrim(featout)
                loss = discrim_criterion( px_r,px_g,px_b ,label_image)
                

                loss.backward()
                adv_loss += loss*spurious_image.shape[0]
                adv_count += spurious_image.shape[0]
                optimizer_discriminator.step()
            print(f"The average adverserial loss for the epoch is {adv_loss/adv_count}")

            

        for spurious_image,label_image,label in tqdm(colored_dataloader_train):
            
            spurious_image,label_image,label = spurious_image.to(device),label_image.to(device),label.to(device)


            optimizer_target.zero_grad()
            optimizer_discriminator.zero_grad()
            
            featout, logits = model(spurious_image)
            target_loss = criterion(logits,label)
            px_r,px_g,px_b = discrim(featout.detach())
            
            maxent_loss = -torch.mean(torch.sum(px_r*torch.log(px_r+1e-16),dim=1))
            maxent_loss += -torch.mean(torch.sum(px_g*torch.log(px_g+1e-16),dim=1))
            maxent_loss += -torch.mean(torch.sum(px_b*torch.log(px_b+1e-16),dim=1))
            
            loss = 0.9*target_loss + 0.1*maxent_loss
            loss.backward()
            optimizer_target.step()

            accuracy += (torch.argmax(logits,dim=1) == label).sum()
            sum_loss += loss*spurious_image.shape[0]
            total_count += spurious_image.shape[0]

        print(f"The average loss for the epoch is {sum_loss/total_count}")
        print(f"The train accuracy for the epoch is {accuracy / total_count}")

        #testing
        model.eval()
        with torch.no_grad():
            accuracy,count  = 0,0

            for spurious_image,label_image,label in colored_dataloader_test:
                spurious_image,label = spurious_image.to(device),label.to(device)
                feat_out,px = model(spurious_image)
                accuracy += (torch.argmax( px ,dim=1) == label).sum()
                count += label.shape[0]
        print(f"The test accuracy for the epoch {epoch}  is {accuracy / count}")
            
   
            
            
        