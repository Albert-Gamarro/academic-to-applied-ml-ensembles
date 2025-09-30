# From Academic to Applied ML Ensembles  

This project transforms academic learning into applied machine learning workflows. Starting from a course-based Jupyter notebook **Course Reference:** *Applied Machine Learning: Ensemble Learning* by Matt Harri. It builds a modular, reusable experimentation framework for ensemble modeling. The goal is to bridge the gap between learning and real-world ML application, emphasizing structured workflows, reproducibility, and actionable insights.  

It serves both as a reference for understanding ensemble techniques and as a working foundation for collaborative data science projects.  

## Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Why This Project Exists](#why-this-project-exists)
- [Scope](#scope)
- [Project Topics](#project-topics)
- [Key Learnings](#key-learnings)

### 🔹 Project Overview  

While the course teaches ensemble ML models through the `ensembles.ipynb` notebook, this project focuses on structuring experimentation into reusable, modular workflows inside the `src/` folder. The goal is to transform ad-hoc exercises into a process that supports efficient, informed decision-making in real-world ML projects.  

The notebook is kept as the reference learning file, while the `src/` scripts are designed to support a real ML lifecycle.  

---

### 🔹 Project Structure  

This repository is organized to clearly separate learning exploration from structured experimentation, providing both a reference and a scalable workflow.  


**Key highlight:**  
The `ensembles.ipynb` notebook was the genesis of this project — it contains the original course material and experiments. From this single learning file, the project evolved into a structured framework in the `src/` folder, demonstrating the process of turning academic exercises into a reproducible ML workflow.  

---

### 🔹 Why This Project Exists  

Machine learning projects require more than isolated experiments — they need structure, reproducibility, and scalability. This project exists to bridge the gap between academic learning and applied ML workflows. Specifically, it aims to:  

- Transform educational notebooks into a structured experimentation framework.  
- Ensure repeatable, modular processes that can handle multiple models, datasets, and parameter configurations.  
- Provide clear analysis of model performance and stability, supporting robust model selection.  
- Enable team-level efficiency: other analysts or data scientists can adopt the workflow without starting from scratch.  

At its core, this project reflects the process of turning knowledge into practice — a fundamental skill for any data science professional who wants to move from theory to applied machine learning.  

---

### 🔹 Scope  

While the full ML lifecycle includes deployment, monitoring, and operational pipelines, this project focuses on the experimentation and evaluation phase, ensuring decisions are informed, repeatable, and scalable.  
.
├── ensembles.ipynb # Original course notebook — the starting point for the project.
├── src/ # Modular scripts for experimentation and evaluation.
│ ├── data/ # Scripts for loading and processing datasets.
│ ├── models/ # Training scripts, model selection, and hyperparameter tuning.
│ └── visualization/ # Visualization scripts for result analysis.
├── config/ # Configuration files for experiments.
├── data/ # Example datasets used for experimentation.
├── models/ # Best trained models with optimal hyperparameters.
├── notebooks/ # Supporting notebooks for hyperparameter analysis, model selection, and overfitting analysis.
├── results/ # Outputs of experiments and visualizations.
├── environment.yml # Environment configuration.
├── setup.py # Project installation script.
└── README.md # Project documentation.

---

### 🔹 Project Topics  

**Modular Approach**  
- **Dataset Loading & Preprocessing:** Standardized scripts to handle initial dataset preparation.  
- **train_model_dev:** Quick experimentation with single models.  
- **train_model_dev_multi:** Multi-model evaluation for comparative analysis across ensemble techniques (bagging, boosting, stacking). Includes stacking as an experiment, following professional practice — tested but only adopted if outperforming individual models.  
- **train_model_prod:** Production-oriented training and evaluation with train/test splits, ensuring reproducibility and stability.  

**Results Analysis & Key Insights**  
- Evaluated metrics and standard deviations to understand both performance and stability.  
- Overfitting diagnostics: checked train-test score gaps for the main metric (F1). Models with large gaps were excluded, ensuring only generalizable winners are considered.  
- Heatmaps allow rapid visualization of best-performing models across metrics and hyperparameter combinations.  

**Why it matters:**  
In production, high average scores alone are not enough. Models with low variability (std) and low overfitting are more reliable and reduce operational risk.  

**Example Insight:**  
From the adult dataset, models winning in F1-score and accuracy were strongest overall. Due to dataset imbalance, F1-score was prioritized — balancing precision and recall for business-relevant decisions.  

---

### 🔹 Key Learnings  

- How to structure ad-hoc academic work into modular, reproducible ML workflows.  
- Effective hyperparameter tuning strategies for ensemble methods (bagging, boosting, stacking).  
- Balancing performance metrics with stability and overfitting diagnostics.  
- The importance of modular project architecture for collaboration and scalability.  
