import torch
import torch.nn as nn
class BelkiCNN(nn.Module):
    def __init__(self,numClasses=5):
        super().__init__()
        self.features= nn.Sequential(
            #блок 1
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True), # inplace true не создает нового тензора, а преобразует имеющийся
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            #блок 2
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            #блок 3
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # последний блок
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128,numClasses)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.pool(x)
        x=self.classifier(x)
        return x
    

def trainOneEpoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss=0
    correct=0
    total=0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss +=loss.item()*images.size(0)
        _, pred= outputs.max(1)
        correct+=pred.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total


def validate(model,dataloader,criterion,device):
    model.eval()
    running_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            running_loss +=loss.item()*images.size(0)
            _,pred= outputs.max(1)
            correct+=pred.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss/total, correct/total

def getPredictions(model,dataloader,device):
    model.eval()
    allPreds=[]
    allLabels=[]
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            preds=outputs.argmax(dim=1)
            allPreds.extend(preds.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
    return allLabels,allPreds

    
    