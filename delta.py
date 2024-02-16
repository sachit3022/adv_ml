"""
This file is to run the experiment to find the adversarial examples, and probe the question of wheather train or test data is more vulnerable to adversarial examples.
Experiment design
1. Get the pre-trained model ( ResNet50, VGG16, MLPMixer, ViT)
2. Get the train and test data of CIFAR10
3. Find average delta for train and test data
4. Compare the average delta for train and test data ( under various norm constraints )
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from torchvision import datasets,transforms
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import cvxpy as cp


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

class Attack:
    def __init__(self,model,device,loss_fn,proj_fn,eps,lr,input_shape):
        self.model = model
        self.model.eval()
        self.device = device
        self.delta = torch.rand(input_shape).to(device)
        self.delta.requires_grad = True
        self.loss_fn = loss_fn
        self.proj_fn = proj_fn
        self.lr = lr
        self.eps = eps
    def __call__(self,x,y):
        self.delta.grad = None
        self.model.zero_grad()
        logit = self.model(x+self.delta)
        loss = self.loss_fn(logit,y)
        loss.backward()
        with torch.no_grad():
            self.delta  -= self.lr*self.delta.grad
            self.delta  = self.proj_fn(self.delta,self.eps,device=self.device)
        return  x + self.delta
    



class TargetedAttack(Attack):
    def __init__(self,model,target_class,device,proj_fn,eps,lr,input_shape):
        super(TargetedAttack,self).__init__(model,device,self.loss_fn,proj_fn,eps,lr,input_shape)  
        self.target_class = target_class
        self.criteria = nn.CrossEntropyLoss()
    def loss_fn(self,logits,y):
        #shape of logits is (batch_size, num_classes)
        y = self.target_class* torch.ones(logits.shape[0]).to(self.device).long()
        return self.criteria(logits,y)
    def attack_accuracy(self,x):
        logits = self.model(x+self.delta)
        return (torch.argmax(logits, dim=1) == self.target_class).float().mean().item()

    

################# Dataset configuration ####################
BATCH_SIZE = 128
NUM_WORKERS = 4

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class Normalise:
    def __init__(self,mean,std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def get_norm(self):
        return transforms.Normalize(mean=self.mean, std=self.std)
    def get_denorm(self):
        return lambda x: x*self.std.view(1,3,1,1) + self.mean.view(1,3,1,1)
Normalisaiton = Normalise([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

################# Model configuration ####################


LR = 1
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

Tau = 0.2

linf_eps = 0.2
l1_eps = 150
l2_eps = 8


################# Projection function tool kit ####################



def proj_2_norm(delta,eps,device=None):
    u = delta.norm(2).item() /  eps
    
    if u > 1:
        delta /= u

    return delta
        
def proj_inf_norm(delta,eps,device=None):
    return delta.clamp_(-eps,eps)


def proj_1_norm(delta,eps,device=None):

    old_shape=  delta.shape
    delta = delta.flatten().cpu().detach().numpy()
    x = cp.Variable(delta.shape)
    objective = cp.Minimize(cp.norm(x - delta, 2)**2)
    constraints = [cp.norm(x, 1) <= eps]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    delta = torch.tensor(x.value,device=device).view(old_shape).type(torch.cuda.FloatTensor)
    delta.requires_grad = True
    return delta

##############################################


if __name__ == "__main__":

    transforms = transforms.Compose([transforms.ToTensor(),Normalisaiton.get_norm()]) #transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
    trainset = datasets.CIFAR10(root='./data', train=True, download=False,transform=transforms)
    testset = datasets.CIFAR10(root='./data', train=False, download=False,transform=transforms)
    valset, testset = torch.utils.data.random_split(testset, [5000, 5000])

    
    
    cifar_data = {'train': DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True),
                'val':  DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True),
                'test': DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True)  }
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(device)
    model.eval()
    

    denorm = Normalisaiton.get_denorm()
    target_class = 7  # cat
    attacks = {'l2': TargetedAttack(model,target_class,device,proj_2_norm,l2_eps,lr=LR,input_shape=(3,32,32)),
                'linf': TargetedAttack(model,target_class,device,proj_inf_norm,linf_eps,lr=LR,input_shape=(3,32,32)),
                'l1': TargetedAttack(model,target_class,device,proj_1_norm,l1_eps,lr=LR,input_shape=(3,32,32))}


    for attack_name,adv_attack in attacks.items():
        for dataset in tqdm(cifar_data['train']):
            x,y = dataset
            x,y = x.to(device), y.to(device)
            adv_attack(x,y)
        
        acc = 0
        for dataset in cifar_data['test']:
            x,y = dataset
            x,y = x.to(device), y.to(device)
            acc += adv_attack.attack_accuracy(x)

        
        ## Tests to check if the attack is working
        print(acc/len(cifar_data['test']))
        print( adv_attack.delta.norm(1), adv_attack.delta.norm(2), adv_attack.delta.norm(float('inf')))
        print(adv_attack.delta.shape)
        fig,ax = plt.subplots(2,1)
        ax[0].imshow(denorm(adv_attack.delta.detach().cpu()).squeeze().permute(1,2,0))
        ax[1].hist(adv_attack.delta.detach().cpu().numpy().flatten(), bins=100)
        fig.savefig(f"results/{target_class}_{attack_name}.png")