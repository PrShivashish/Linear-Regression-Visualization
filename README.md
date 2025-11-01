# 📊 Linear Regression From Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Latest-yellow?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Latest-purple?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=flat-square&logo=matplotlib)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

### 🚀 Build Machine Learning Fundamentals with Pure Python & NumPy

**Master the Mathematics Behind Linear Regression Through Implementation**

[📖 Documentation](#-implementation-details) • [🎯 Features](#-key-features) • [⚙️ Installation](#-installation--setup) • [💻 Usage](#-usage--quick-start) • [🔍 Results](#-results--visualizations)

</div>

---

## 🎯 Project Overview

This repository contains a comprehensive implementation of **Linear Regression from scratch** using only NumPy and Python. Instead of relying on scikit-learn or TensorFlow, this project builds the algorithm from the ground up—implementing cost functions, gradient descent optimization, feature scaling, and dynamic visualizations. Perfect for understanding the mathematical foundations of machine learning.

### ✨ Why This Repository?

- 🧠 **Deep Learning**: Understand the mathematics behind linear regression
- 🔧 **Pure Implementation**: Built with NumPy—no black-box ML libraries
- 📈 **Feature Scaling**: Includes normalization techniques for better convergence
- 🎬 **Interactive Animations**: Visualize gradient descent in real-time
- 📊 **Performance Analysis**: Cost function tracking and convergence visualization
- 🎓 **Educational Value**: Perfect for ML beginners and interview preparation

---

## 🎯 Key Features

✅ **Single-Feature Linear Regression**
   - Predict salary based on years of experience
   - Simple yet comprehensive implementation

✅ **Gradient Descent Algorithm**
   - Manual implementation of gradient descent optimization
   - Configurable learning rate and iterations
   - Cost function minimization tracking

✅ **Feature Scaling & Normalization**
   - Z-score standardization implementation
   - Improved convergence speed
   - Faster training compared to unscaled data

✅ **Dynamic Visualizations**
   - Scatter plots with regression lines
   - Cost vs. iterations graphs
   - Animated gradient descent process (MP4 export)
   - Real-time model fitting visualization

✅ **Mathematical Rigor**
   - Mean Squared Error (MSE) calculation
   - Partial derivatives for parameter updates
   - Loss function convergence analysis

---

## 📋 Project Structure

```
linear-regression-from-scratch/
├── Linear_regression(Single_feature)1.ipynb  # Main Jupyter Notebook
├── experience_salary_dataset.xlsx             # Sample Dataset (Experience vs Salary)
├── gradient_descent_animation.mp4             # Generated Animation
├── README.md                                  # This file
└── requirements.txt                           # Dependencies

```

### 📁 Dataset Details

**File**: `experience_salary_dataset.xlsx`

| Column | Description | Range |
|--------|-------------|-------|
| Experience (Years) | Years of professional experience | 1-20 years |
| Salary (USD) | Annual salary in USD | ~$100K-$130K+ |

---

## 🔧 Implementation Details

### 1️⃣ **Core Functions Implemented**

#### Prediction Function
```
ŷ = w·X + b
```
Where:
- `w` = Weight (slope)
- `X` = Input feature
- `b` = Bias (y-intercept)

#### Cost Function (Mean Squared Error)
```
J(w,b) = (1/2m) Σ(ŷᵢ - yᵢ)²
```
Where:
- `m` = Number of training samples
- `ŷᵢ` = Predicted value
- `yᵢ` = Actual value

#### Gradient Descent Updates
```
w = w - α · (∂J/∂w)
b = b - α · (∂J/∂b)
```
Where:
- `α` = Learning rate
- `∂J/∂w`, `∂J/∂b` = Partial derivatives

### 2️⃣ **Feature Scaling (Standardization)**

```
X_scaled = (X - μ) / σ
```
Where:
- `μ` = Mean of feature
- `σ` = Standard deviation

**Benefits**:
- Accelerates convergence
- Prevents numerical instability
- Makes learning rate independent of feature scale

### 3️⃣ **Algorithm Workflow**

```
1. Initialize w = 0, b = 0
2. Set hyperparameters (learning_rate, iterations)
3. FOR each iteration:
   - Calculate predictions: ŷ = w·X + b
   - Compute cost: J(w,b)
   - Calculate gradients: dw, db
   - Update parameters: w, b
   - Store cost history
4. Return optimized w, b
```

---

## 📊 Results & Visualizations

### 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Optimal Weight (w)** | ~6591.17 |
| **Optimal Bias (b)** | ~12732.37 |
| **Final MSE (Unscaled)** | Minimized |
| **Convergence** | 1000 iterations |
| **Learning Rate** | 0.01 |

### 📉 Generated Visualizations

1. **Manual Parameter Testing**: Visual comparison of different (w, b) pairs
2. **Cost Function Decay**: Shows how cost decreases with iterations
3. **Regression Line Fit**: Final model overlaid on actual data
4. **Scaled vs. Unscaled**: Performance comparison
5. **Animated Gradient Descent**: MP4 video of parameter updates in real-time

---

## 💻 Installation & Setup

### ✅ Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **pip** (Python package manager)
- **Git** (to clone the repository)

### 🚀 Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/uroojzarab/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

#### 2. Create Virtual Environment (Recommended)
```bash
# On Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas matplotlib ipython openpyxl
```

#### 4. Install FFmpeg (for animation export)

- **Windows**: [Download](https://ffmpeg.org/download.html) and add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install ffmpeg`

Verify installation:
```bash
ffmpeg -version
```

#### 5. Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `Linear_regression(Single_feature)1.ipynb` in your browser.

---

## 📖 Usage & Quick Start

### 🎮 Running the Notebook

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Linear_regression(Single_feature)1.ipynb
   ```

2. **Update Data Path** (if needed):
   ```python
   # Change this line to your local data path
   data = pd.read_excel("experience_salary_dataset.xlsx")
   ```

3. **Run Cells Sequentially**:
   - Click `▶ Run` button or press `Shift + Enter`
   - Execute cells in order to avoid errors

4. **View Results**:
   - Visualizations appear inline
   - MP4 animation saves to project directory

### 📝 Example Code Snippet

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel("experience_salary_dataset.xlsx")
X = data['Experience (Years)'].values
Y = data['Salary (USD)'].values

# Initialize parameters
w, b = 0, 0
alpha = 0.01
iterations = 1000

# Gradient descent
for i in range(iterations):
    # Predictions
    y_pred = w * X + b
    
    # Cost
    cost = np.mean((y_pred - Y) ** 2) / 2
    
    # Gradients
    dw = np.sum((y_pred - Y) * X) / len(X)
    db = np.sum(y_pred - Y) / len(X)
    
    # Update
    w -= alpha * dw
    b -= alpha * db

print(f"Optimal w: {w:.2f}")
print(f"Optimal b: {b:.2f}")

# Predict
y_predict = w * X + b
plt.scatter(X, Y, label='Actual')
plt.plot(X, y_predict, 'r-', label='Predicted')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (USD)')
plt.legend()
plt.show()
```

---

## 📊 Hyperparameter Tuning

Customize the learning process by adjusting these parameters in the notebook:

```python
learning_rate = 0.01      # Controls step size (0.001 - 0.1 recommended)
iterations = 1000         # Number of gradient descent steps
alpha = 0.01             # Learning rate alias
m = len(X)               # Number of training samples
```

### 💡 Hyperparameter Tips

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Learning Rate (α)** | 0.0001 - 1.0 | Higher = faster but may diverge; Lower = slower but stable |
| **Iterations** | 100 - 10000 | More = better fit but slower; Fewer = faster but underfitting |

---

## 🎓 Learning Outcomes

After working through this repository, you will understand:

✅ How linear regression works mathematically  
✅ Cost function (Mean Squared Error) fundamentals  
✅ Gradient descent optimization algorithm  
✅ Feature scaling and normalization techniques  
✅ How to build ML models from scratch  
✅ Parameter initialization and tuning  
✅ Model evaluation and visualization  

---

## 🔍 Key Concepts Explained

### 📌 Gradient Descent

Iteratively moves parameters towards the minimum of the cost function using gradients (slopes) as direction guides.

### 📌 Cost Function

Measures the difference between predicted and actual values. Lower cost = better fit.

### 📌 Feature Scaling

Transforms features to similar scales, preventing larger-magnitude features from dominating the learning process.

### 📌 Convergence

The process where the cost function stops decreasing significantly, indicating the algorithm has found optimal parameters.

---

## 📈 Performance Comparison

### Unscaled vs. Scaled Data

| Aspect | Unscaled | Scaled |
|--------|----------|--------|
| **Convergence Speed** | Slower | ⚡ Faster |
| **Weight Range** | Large (~6591) | Moderate (~37437) |
| **Numerical Stability** | Less stable | More stable |
| **Learning Rate** | Sensitive | Less sensitive |

---

## 🛠️ Troubleshooting

### ❌ Problem: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Run `pip install pandas` in your activated virtual environment.

### ❌ Problem: "FileNotFoundError: experience_salary_dataset.xlsx not found"
**Solution**: Verify the dataset path in the notebook matches your file location.

### ❌ Problem: "FFmpeg not found" (animation export fails)
**Solution**: Install FFmpeg and add it to your system PATH.

### ❌ Problem: Notebook kernel crashes
**Solution**: Restart the kernel (Kernel > Restart) and run cells sequentially.

---

## 📚 Resources & References

- [Linear Regression Mathematics](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html)
- [Gradient Descent Explained](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Visualization Guide](https://matplotlib.org/)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for personal and commercial purposes.

---

## 👨‍💻 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs via GitHub Issues
- 💡 Suggest improvements
- 🔀 Submit pull requests with enhancements
- 📝 Improve documentation

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 🎉 Use Cases

This repository is perfect for:

- 📚 **Machine Learning Students**: Learn ML fundamentals
- 👔 **Job Interviews**: Demonstrate algorithm understanding
- 🔬 **Research**: Baseline implementation for experiments
- 🎓 **Teaching**: Educational material for data science courses
- 💼 **Portfolio Projects**: Showcase coding and ML knowledge

---

## 🌟 Featured By

- Machine Learning Education Communities
- Data Science Learning Platforms
- GitHub Awesome Lists

---

## 📞 Support & Questions

Have questions or suggestions? Feel free to:

- 📧 Open an GitHub Issue
- 💬 Start a Discussion
- 🔔 Watch for updates

---

## ⭐ Show Your Support

If this repository helped you understand linear regression better, please consider:

- ⭐ **Starring** the repository
- 🔖 **Bookmarking** for future reference
- 👥 **Sharing** with others learning ML
- 💬 **Leaving feedback** in the issues

---

<div align="center">

### Made with ❤️ by [Shivashish Prusty](https://github.com/PrShivashish)

**Happy Learning! Keep Building! 🚀**

[⬆ Back to Top](#-linear-regression-from-scratch)

</div>
