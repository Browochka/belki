import torch
import os
import csv
import random
import yaml
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from preparing import BelkiDataset, trainTransform, testTransform
from modeling import BelkiCNN,trainOneEpoch,validate,getPredictions
from makeplots import makeAcc,makeConfuse,makeLoss


os.makedirs("models",exist_ok=True)
os.makedirs("metrics",exist_ok=True)


params= yaml.safe_load(open("params.yaml"))
device = torch.device("mps")
numEpochs=params["train"]["epochs"]
batchSize=params["train"]["batchSize"]

TrainDataset = BelkiDataset("data/processed/train.csv",transform=trainTransform)
TestDataset = BelkiDataset("data/processed/test.csv",transform=testTransform)

trainLoader = DataLoader(TrainDataset,batch_size=batchSize,shuffle=True)
testLoader = DataLoader(TestDataset,batch_size=batchSize,shuffle=False)

cnn=BelkiCNN(numClasses=5).to(device)
resnet=models.resnet34(pretrained=True)
resnet.fc=nn.Linear(resnet.fc.in_features,5)
for param in resnet.parameters():
    param.requires_grad = False  
for param in resnet.fc.parameters():
    param.requires_grad = True
resnet = resnet.to(device)

modelis=[resnet,cnn]
summary={}
criterion = nn.CrossEntropyLoss()

for model in modelis:
    history = {
    "trainLoss": [],
    "testLoss": [],
    "trainAcc": [],
    "testAcc": []}
    testbest=0
    trainbest=0
    if isinstance(model, BelkiCNN):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    for epoch in range(numEpochs):
        trainloss, trainacc = trainOneEpoch(model, trainLoader, criterion, optimizer, device)
        testloss, testacc = validate(model, testLoader, criterion, device)
        trainbest=max(trainbest,trainacc)
        testbest=max(testbest,testacc)
        history["trainLoss"].append(trainloss)
        history["testLoss"].append(testloss)
        history["trainAcc"].append(trainacc)
        history["testAcc"].append(testacc)
    print(f"finish of {type(model).__name__}")
    summary[type(model).__name__] = {
        "best on train": trainbest,
        "best on test": testbest}
    makeLoss(history["trainLoss"],history["testLoss"],type(model).__name__)
    makeAcc(history["trainAcc"],history["testAcc"],type(model).__name__)
    torch.save(model.state_dict(), f'models/{type(model).__name__}.pth')



labels, preds = getPredictions(resnet, testLoader, device)
cm = confusion_matrix(labels, preds)
makeConfuse(cm,type(resnet).__name__)



with open("metrics/summary.json", "w") as f:
    json.dump(summary, f, indent=4)