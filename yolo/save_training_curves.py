import numpy as np
import csv
from ultralytics import YOLO

'''
This is a folder that runs the best trained YOLO model after yolo_train.py and then
spits out a CSV containing all of the data necessary to construct the output validation
graphs, such as PR curves, F1-confidence curves, confusion matrices, etc.
'''

# Load model and validate
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="configs.yaml")

# Loop through curves
for name, arr in zip(metrics.curves, metrics.curves_results):
    x_vals, y_vals, x_label, y_label = arr  # unpack
    clean_name = name.replace('(', '').replace(')', '').replace('-', '_').replace(' ', '_')
    csv_filename = f"{clean_name}.csv"
    
    # Make sure x_vals and y_vals are 1D arrays
    x_vals = np.array(x_vals).flatten()
    y_vals = np.array(y_vals).flatten()
    
    # Write to CSV
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([x_label, y_label])  # header
        for x, y in zip(x_vals, y_vals):
            writer.writerow([x, y])
    
    print(f"Saved {csv_filename}")

# Save confusion matrix as CSV
cm = metrics.confusion_matrix.matrix
np.savetxt("confusion_matrix.csv", cm, delimiter=",", fmt='%d')
print("Saved confusion_matrix.csv")
