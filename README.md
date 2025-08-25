<h1 align="center">
  iCAM-Net
</h1>
<p align="center">
An Interpretable Cross-Attention based Multi-task framework for predicting Herb-Disease Associations
</p>

## 📖 Overview

`iCAM-Net` is a deep learning model based on graph neural networks, designed to predict potential associations between herbs and diseases. The model leverages a multi-task learning framework to simultaneously learn **Herb-Disease (HD)** association prediction and **Compound-Protein (CP)** interaction prediction, thereby enhancing the primary task's performance and the model's interpretability.

---
## 📂 Dataset

The dataset for this project includes information on herbs, chemical compounds, diseases, and their associated target proteins. All data files should be placed in the `data/` directory.

#### Recommended Directory Structure
The code assumes the following project structure by default:
```

iCAM-Net/
├── data/
│   ├── HERB/               # HERB dataset
│   │   ├── H_C_HERB.csv
│   │   ├── D_P_HERB.csv
│   │   ├── H_D_HERB.csv
│   │   ├── H_D_HERB_neg.csv
│   │   ├── C_P_HERB.csv
│   │   ├── C_P_HERB_neg.csv
│   │   ├── ce_HERB.txt
│   │   └── pe_HERB.txt
│   ├── H_C_TCM.csv
│   ├── D_P_TCM.csv
│   ├── H_D_TCM.csv
│   ├── H_D_TCM_neg.csv
│   ├── C_P_TCM.csv
│   ├── C_P_TCM_neg.csv
│   ├── ce_TCM.txt
│   └── pe_TCM.txt
├── code/
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   ├── graph.py
│   ├── dataset.py
│   └── HGNN.py
└── requirements.txt

```
> **Note:** The original data can be obtained from relevant sources and should be organized according to the structure above.

---
## 🚀 Installation

First, clone this repository to your local machine. We strongly recommend using a virtual environment to manage project dependencies.
```
git clone https://github.com/qunshanxingyun/iCAM-Net.git
cd iCAM-Net
```

1.  **Create and Activate a Virtual Environment (Optional but Recommended)**
    ```bash
    conda create -n icam python=3.12
    conda activate icam
    ```

2.  **Install Dependencies**
    All required packages are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---
## ⚡️ Running the Model

All training and evaluation processes can be initiated by running the `main.py` script.

1.  **Navigate to the `code` directory**
    ```bash
    cd code
    ```

2.  **Execute the main script**
    ```bash
    python main.py
    ```
    The script will automatically load the data, build the model, split the datasets, and start the training and evaluation process.

---
## ⚙️ Configuration

Hyperparameters, file paths, and other configurations can be modified directly within the `main` function in `code/main.py`.

---

## ⚡️ Training the Model

All training and evaluation processes can be initiated by running the `main.py` script.

1.  **Navigate to the `code` directory**:
    ```bash
    cd code
    ```

2.  **Modify Configuration (Optional)**:
    Hyperparameters, file paths, and other configurations can be adjusted directly within the `main` function in `code/main.py`.

3.  **Execute the main script**:
    ```bash
    python main.py
    ```
    The script will load the data, build the model, split the datasets, and start the training and evaluation pipeline.

---

## 🔬 Inference

We provide a dedicated Jupyter Notebook.

**File**: `code/inference_and_case_study.ipynb`

To use it, simply open the notebook in a Jupyter environment and follow the instructions in each cell.

---

## 📈 Results & Tracking

- **Local Results**: During training, the best-performing model (based on validation loss) will be saved to a timestamped directory inside `result/`. Final test results are printed to the console.
- **Wandb**: We use `wandb` (Weights & Biases) for experiment tracking. If you have a `wandb` account, log in before running, and all metrics will be automatically tracked. To disable this, set `use_wandb=False` in the `MultiTaskTrainer` initialization in `main.py`.


---
