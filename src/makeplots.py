import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


#classes = {"belka_finlay":0, "belka_karoling":1, "deppe":2, "gimalay_belka":3, "prevost":4}
os.makedirs("plots",exist_ok=True)
params= yaml.safe_load(open("params.yaml"))
EpochRange=[i+1 for i in range(params["train"]["epochs"])]

def makeConfuse(cm,modname):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["belka_finlay", "belka_karoling","deppe","gimalay_belka","prevost"],  
    yticklabels=["belka_finlay", "belka_karoling","deppe","gimalay_belka","prevost"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix {modname}")
    plt.tight_layout()
    plt.savefig(f"plots/confusionMatrix{modname}.png")
    plt.close()

def makeLoss(trainLoss,testLoss,modname):
    plt.figure()
    plt.plot(EpochRange, trainLoss)
    plt.plot(EpochRange, testLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {modname}")
    plt.legend(["Train Loss", "Test Loss"])
    plt.savefig(f"plots/{modname}Loss.png")
    plt.close()


def makeAcc(trainAcc,testAcc,modname):
    plt.figure()
    plt.plot(EpochRange, trainAcc)
    plt.plot(EpochRange, testAcc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {modname}")
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.savefig(f"plots/{modname}_accuracy.png")
    plt.close()
    