import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# ================== CONFIG ================== #
SOURCE_ROOT = r"E:\Upwork Project\AI_Leak_Detection_Project\data\processed"
DEST_ROOT   = r"E:\Upwork Project\AI_Leak_Detection_Project\images\cwt_log_new"

SENSORS = ["Accelerometer", "Dynamic Pressure Sensor", "Hydrophones"]
CLASSES = ["No-leak", "Orifice Leak", "Gasket Leak", "Longitudinal Crack", "Circumferential Crack"]
MODE_SUBDIR = "Looped"

WAVELET = "morl"
SCALES = np.arange(1, 128)   # adjust if needed
IMG_SIZE = (2.56, 2.56)      # inches → 256x256 pixels (dpi=100)
COLUMN = "Value"
# ============================================ #

def save_cwt_image(signal, save_path):
    # normalize signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    coefs, _ = pywt.cwt(signal, SCALES, WAVELET)
    power = np.abs(coefs) ** 2

    plt.figure(figsize=IMG_SIZE, dpi=100)
    plt.imshow(power, aspect="auto", cmap="jet", norm=LogNorm())
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    for sensor in SENSORS:
        for klass in CLASSES:
            in_dir  = os.path.join(SOURCE_ROOT, sensor, MODE_SUBDIR, klass)
            out_dir = os.path.join(DEST_ROOT, sensor, MODE_SUBDIR, klass)
            os.makedirs(out_dir, exist_ok=True)

            print(f"📂 Processing {sensor} - {klass}")
            for file in tqdm([f for f in os.listdir(in_dir) if f.endswith(".csv")]):
                csv_path = os.path.join(in_dir, file)
                signal = pd.read_csv(csv_path)[COLUMN].values
                out_path = os.path.join(out_dir, file.replace(".csv", ".png"))
                save_cwt_image(signal, out_path)

    print("✅ All CWT Log images generated successfully.")

if __name__ == "__main__":
    main()
