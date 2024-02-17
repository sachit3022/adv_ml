from dataset import WholeDataLoader
from dotted_dict import DottedDict
from torch.utils.data import DataLoader,dataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


LR = 3e-4
if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

if __name__ == "__main__":
    
    option_train  = DottedDict({"data_dir":"data/colored_mnist", "color_var":0.020,"data_split":"train"})
    option_test  = DottedDict({"data_dir":"data/colored_mnist", "color_var":0.020,"data_split":"test"})
        
    
    colored_minist_dataset_train = WholeDataLoader(option_train)
    colored_minist_dataset_test = WholeDataLoader(option_test)

    
    colored_dataloader_train = DataLoader(colored_minist_dataset_train, num_workers= 8, persistent_workers = True,batch_size = 1024,shuffle=True)
    colored_dataloader_test = DataLoader(colored_minist_dataset_test, num_workers= 8, persistent_workers = True,batch_size = 1024,shuffle=False)
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = LR)

    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)


    for epoch in range(2):
        model.train()
        sum_loss,total_count = 0,0
        for spurious_image,true_image,label in tqdm(colored_dataloader_train):
            spurious_image,label = spurious_image.to(device),label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(spurious_image),label)
            loss.backward()
            optimizer.step()
            sum_loss += loss*spurious_image.shape[0]
            total_count += spurious_image.shape[0]
        print(f"The averge loss for the epoch is {sum_loss/total_count}")

        #testing
        model.eval()
        with torch.no_grad():
            accuracy,count  = 0,0

            for spurious_image,true_image,label in colored_dataloader_test:
                spurious_image,label = spurious_image.to(device),label.to(device)
                accuracy += (torch.argmax(model(spurious_image),dim=0) == label).sum()
                count += label.shape[0]
        prinf(f"The test accuracy for the epoch {epoch}  is {accuracy / count}")
            
            
            
            
            