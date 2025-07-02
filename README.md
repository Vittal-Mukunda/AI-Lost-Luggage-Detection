# 🛄 FindMyBag — AI Lost Bag Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Powered by CLIP](https://img.shields.io/badge/Model-OpenAI%20CLIP-green.svg)](https://github.com/openai/CLIP)

---

**FindMyBag** is an AI-powered desktop application that helps locate lost luggage at airports by visually matching a user-uploaded image with previously seen bags from CCTV or surveillance footage.

---

## ✨ What It Does

FindMyBag uses OpenAI's CLIP model to understand the visual features of bags and FAISS to search through a dataset of known luggage images. It then shows the best match (if any) based on image similarity.

- Upload a photo of the lost bag
- AI analyzes and compares it to stored surveillance images
- Displays the closest matching result with a similarity score
- Automatically exports match results to a folder
- Simple, user-friendly interface with no internet required after setup

---

## 🧠 How It Works

- **CLIP (ViT-L/14@336px)** extracts high-dimensional features from each bag image
- **FAISS** performs fast similarity search using vector distance
- **Tkinter GUI** provides an intuitive drag-and-drop interface
- **Local Output** includes query and match images + a result text file

---

## 📸 Screenshots (optional)
*Coming soon — example query/match interface*

---

## 📁 Folder Structure

