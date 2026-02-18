import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import gradio as gr
from preparing import testTransform

belki=["belka_finlay", "belka_karoling","deppe","gimalay_belka","prevost"]
model = models.resnet34(weights=None)
model.fc =nn.Linear(model.fc.in_features,5)
model.load_state_dict(torch.load("models/ResNet.pth",weights_only=True))
model.eval()

def classification(img):
    img=Image.fromarray(img)
    tenzor = testTransform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tenzor)
        _, predicted = torch.max(outputs, 1)
    return belki[predicted.item()]

inter= gr.Interface(
    fn=classification,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="Угадываем белочек"
)

inter.launch()



