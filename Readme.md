
# SoyScan 🌱 – Interpretable Soybean Disease Classification

SoyScan is a Flask-based web application that performs soybean leaf and seed disease classification using a MaxViT model. It includes Grad-CAM heatmap visualizations for interpretability and supports side-by-side upload and prediction preview.

## 🚀 Features

- 📸 Upload image for soybean leaf or seed
- 🧠 Predict disease using MaxViT
- 🔥 Grad-CAM-based visual explanations
- 📊 Probability bar chart with class labels
- 💾 Downloadable CAM heatmaps
- 🎨 Separate UI for leaf and seed prediction
- ⚡ Clean UI with modern gradients and styling

## 📁 Project Structure

```
soybean_app/
├── app.py                  # Main Flask server
├── utils.py                # Grad-CAM, model loading, and prediction
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── index.html          # Landing page
│   ├── leaf.html           # Soybean leaf prediction page
│   └── seed.html           # Soybean seed prediction page
├── static/
│   ├── uploads/            # Uploaded images
│   └── cams/               # Generated Grad-CAM outputs
├── model/
│   └── state of the trained MAXVIT model.txt  # Model notes
└── README.md             
```

## 🧪 How to Use

1. Run the Flask app:
```bash
python app.py
```

2. Visit `http://127.0.0.1:5000`

3. Choose between:
   - 🟢 Soybean Leaf Classification (`/leaf`)
   - 🟠 Soybean Seed Classification (`/seed`)

4. Upload your image, hit Predict, and view:
   - Category probabilities
   - Predicted class
   - Grad-CAM visualization
   - Downloadable heatmap

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/soybean-app.git
cd soybean-app

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

## 📊 Model & Datasets

- 🔍 Backbone: MaxViT (pretrained via `timm`)
- 📂 Input: Leaf/Seed images
- 📁 Output: Disease classes (labels loaded dynamically from trained model)
- 🧪 Grad-CAM: Layer visualized via torch hooks

Model file not included — specify `.pth` location in `utils.py`

## 🖼️ Screenshots & Demo

### 🎥 Demo Video
> `Demo/demo.mp4`

### 🔎 Prediction Example
> Add screenshots showing:
> - Upload form
> - Grad-CAM visualization
> - Prediction results layout

## 📦 Dependencies

- Flask
- PyTorch
- timm
- torchvision
- numpy
- opencv-python
- matplotlib
- Jinja2

## 🙏 Acknowledgements

- MaxViT by Google Research (via timm)
- Grad-CAM techniques
