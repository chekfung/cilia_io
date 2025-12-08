import os 
import numpy as np
from ultralytics import YOLO
import gc
import torch

'''
Fine Tune YOLO Model on the configs.yaml dataset for 300 epochs
'''

gc.collect()

torch.cuda.empty_cache()

# Load Model
model = YOLO("yolo11m.pt")

# Train
epochs = 300
results = model.train(data='configs.yaml', epochs=epochs, batch=4, cache=False)
metrics = model.val()  # evaluate model performance on the validation set

