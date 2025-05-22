# 🌱 PhoTorch


PhoTorch is an open-source tool for plant ecophysiologists and crop modelers, designed to streamline the fitting and analysis of core physiological models from gas exchange and water relations data.

🌐 Web app: Try it out at https://photorch.streamlit.app

🐳 Docker-based desktop app (Recommended): Install and run locally following [instructions](#local-setup).

📄 Accompanying publications: [[PhoTorch](https://doi.org/10.48550/arXiv.2501.15484)]


## 🔬 What does PhoTorch do?

PhoTorch provides an intuitive, interactive platform for fitting widely used plant physiological models, enabling rapid exploration and interpretation of experimental data.


## 🚀 Key Features
- Photosynthesis Model Fitting
  - Fit the full Farquhar-von Caemmerer-Berry (1980) model for:
    - V<sub>cmax25</sub>, J<sub>max25</sub>, TPU<sub>25</sub>, and 16+ others
  - Integrate CO₂ (A–Ci), light response (A–Q), and temperature response (A-T) curves
  - Recover Temperature Optima
- Stomatal Conductance Model Fitting
  - Fit multiple empirical and optimization-based models, including:
  	- Ball-Woodrow-Berry (1987)
  	- Medlyn et al. (2011)
  	- Leuning (1995)
  	- Buckley-Turnbull-Adams (2012)
- Pressure–Volume (PV) Curve Fitting
  - Analyze water relations data to extract parameters such as:
  	- Osmotic potential at full turgor (π<sub>0</sub>)
  	- Bulk modulus of elasticity (ε)
  	- Relative water content at turgor loss point (RWC<sub>tlp</sub>)
- Flexible Data Input
- Accepts .csv, .txt, and .xlsx formats
- Optional rescaling with survey data
- Species-specific analysis and visualization
- Interactive and Visual
- Real-time fitting feedback
- Plotly-based visualization of fit quality
- View RMSE, R², and residuals for model evaluation


## 🧪 Who is it for?

PhoTorch is designed for:
- Field scientists needing a fast way to process LI-600 or LI-6800 data
- Plant ecophysiologists analyzing gas exchange and water relations
- Crop modelers calibrating parameters for predictive models
- Students and educators learning how physiological models work


## 📦 Installation Options

🌍 Use it in the browser
- No installation required
- Try it instantly at https://photorch.streamlit.app
- (Currently compute limited)

🖥️ Run locally via Docker
- Clone this repository and run via Docker for full offline functionality
- See Installation Instructions below
---
<a name="local-setup"></a>
## 🛠️ Local Setup Instructions (All Platforms)

### Step 0: Install Docker

#### macOS
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Install it and launch Docker
3. Allow permissions when prompted

#### Windows
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Follow the installation prompts (Enable **WSL 2** if asked)
3. Launch Docker

#### Linux (Ubuntu/Debian)
Follow instructions [here](https://docs.docker.com/desktop/setup/install/linux/ubuntu/) or try
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

### Step 1: Download the App 
```bash
git clone https://github.com/ktrizzo/photorch-app.git
cd photorch-app
```

### Step 2: Launch the App
```bash
chmod +x launch.sh
./launch.sh
```
If not opened in a web browser tab automatically, go to https://localhost:8501.

### Step 3: Enjoy
Fit lots of models!

### Step 4: Close the App
```bash
docker stop photorch-app
docker rm photorch-app
```

