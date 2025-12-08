from ultralytics import YOLO
import numpy as np
import csv

'''
Run the trained YOLO model from yolo_train.py on the independent test set split.
'''

def main():
    # Load your trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Run evaluation on the test set defined inside data.yaml
    metrics = model.val(
        data="configs.yaml",     # dataset config file
        split="test",          # evaluate on the test split
        save=True
    )
    
    # Print results dictionary (mAP, precision, recall, etc.)
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

if __name__ == "__main__":
    main()
