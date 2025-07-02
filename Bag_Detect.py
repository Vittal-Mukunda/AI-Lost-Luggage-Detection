import os

# Environment & Imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2 as cv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import clip
import numpy as np
import faiss
import shutil

# Configuration
DISTANCE_THRESHOLD = 0.025
OUTPUT_FOLDER = "output" #where matched results (query image, match image, and info) are saved — not the dataset.
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
print("✅ CLIP model loaded.")

# Dataset Loading & Feature Extraction
dataset_folder = r"FOLDERS"
filenames = []
features = []

print("🔍 Scanning dataset...")
for fname in os.listdir(dataset_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image_path = os.path.join(dataset_folder, fname)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image).cpu().numpy()
            features.append(feature[0])
            filenames.append(fname)
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

features = np.array(features).astype('float32')
if len(features) == 0:
    raise ValueError("🚫 No valid images processed. Your 'bags/' folder may be empty or corrupted.")

# Build FAISS Index
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)
print(f"✅ FAISS index built with {len(filenames)} bags.")

# GUI Setup
root = tk.Tk()
root.title("Bag Matcher")
root.geometry("900x600")
root.configure(bg="white")

title = tk.Label(root, text="🎒  Bag Matcher AI", font=("Helvetica", 24, "bold"), fg="#2c3e50", bg="white")
title.pack(pady=20)

frame = tk.Frame(root, bg="white")
frame.pack(pady=20)

query_label = tk.Label(frame, text="QUERY IMAGE", font=("Helvetica", 14, "bold"), fg="#2c3e50", bg="white")
query_label.grid(row=0, column=0, padx=40, pady=10)

match_label = tk.Label(frame, text="MATCHED IMAGE", font=("Helvetica", 14, "bold"), fg="#2c3e50", bg="white")
match_label.grid(row=0, column=1, padx=40, pady=10)

query_canvas = tk.Label(frame, bg="white", relief="solid", bd=1)
query_canvas.grid(row=1, column=0, padx=40, pady=10)

match_canvas = tk.Label(frame, bg="white", relief="solid", bd=1)
match_canvas.grid(row=1, column=1, padx=40, pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 13), fg="#2c3e50", bg="white")
result_label.pack(pady=20)

image_refs = {}

# Core Matching Logic
def show_result(query_img_path):
    try:
        image = preprocess(Image.open(query_img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_vec = model.encode_image(image).cpu().numpy().astype('float32')
        D, I = index.search(query_vec, k=1)
        matched_fname = filenames[I[0][0]]
        distance = D[0][0]

        print(f"Debug - Match Percentage : {distance:.6f}")

        qimg = Image.open(query_img_path).resize((250, 200), Image.Resampling.LANCZOS)
        qtk = ImageTk.PhotoImage(qimg)
        query_canvas.config(image=qtk, width=250, height=200)
        image_refs["query"] = qtk

        match_path = os.path.join(dataset_folder, matched_fname)
        mimg = Image.open(match_path).resize((250, 200), Image.Resampling.LANCZOS)
        mtk = ImageTk.PhotoImage(mimg)
        match_canvas.config(image=mtk, width=250, height=200)
        image_refs["match"] = mtk

        if distance <= DISTANCE_THRESHOLD:
            result_label.config(
                text=f"✅ Match Found: {matched_fname}\n🔍 Match Percentage: {distance:.1f}"
            )
        else:
            result_label.config(
                text=f"⚠️ Closest Match (Low Match Percentage): {matched_fname}\n🔍 Match Percentage: {distance:.1f}"
            )

        shutil.copy(query_img_path, os.path.join(OUTPUT_FOLDER, "query.jpg"))
        shutil.copy(match_path, os.path.join(OUTPUT_FOLDER, "matched.jpg"))
        with open(os.path.join(OUTPUT_FOLDER, "match_info.txt"), "w") as f:
            f.write(f"Query: {os.path.basename(query_img_path)}\n")
            f.write(f"Closest Match: {matched_fname}\n")
            f.write(f"Match Percentage: {distance:.1f}%\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error details: {e}")

# File Picker Button Action
def open_image():
    filepath = filedialog.askopenfilename(
        title="Choose a bag image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if filepath:
        show_result(filepath)

# Upload Button
btn = tk.Button(root, text="📂 Upload Bag Image", command=open_image, font=("Arial", 14), bg="#4CAF50", fg="white",
                padx=10, pady=5)
btn.pack(pady=10)

# Tip Label
debug_label = tk.Label(root, text="💡 Tip: Check console for distance values to fine-tune thresholds",
                       font=("Arial", 10), fg="#7f8c8d", bg="white")
debug_label.pack(pady=5)

# Start GUI Loop
root.mainloop()
