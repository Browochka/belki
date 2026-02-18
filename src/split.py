import os
import csv
import random
import yaml
from sklearn.model_selection import train_test_split


params= yaml.safe_load(open("params.yaml"))
random.seed(params["seed"])

os.makedirs("data/processed", exist_ok=True)


classes = {"belka_finlay":0, "belka_karoling":1, "deppe":2, "gimalay_belka":3, "prevost":4}
print(classes)
rows = []

for label in os.listdir("data/raw"):
    class_dir = os.path.join("data/raw", label)
    if not os.path.isdir(class_dir):
        continue
    for img in os.listdir(class_dir):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            rows.append((os.path.join(class_dir, img), classes[label]))


n = len(rows)

train_rows,test_rows = train_test_split(rows,test_size=params["split"]["testSize"],random_state=params["seed"],stratify=[label for _, label in rows])

with open("data/processed/test.csv","w",newline="", encoding="utf-8") as f:
    writer=csv.writer(f)
    writer.writerow(["imagePath","label"])
    writer.writerows(test_rows)
with open("data/processed/train.csv","w",newline="", encoding="utf-8") as f:
    writer=csv.writer(f)
    writer.writerow(["imagePath","label"])
    writer.writerows(train_rows)

print("CSV для train/test созданы")
