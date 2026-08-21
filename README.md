# Football Player Position Classifier

## Introduction

I built this to see if you can guess a football player's position just from their FIFA/FC stats — no name, no club, nothing, just numbers like pace, passing, defending, etc.

Turns out you can, pretty well.

## Problem Statement

Every player in the dataset has a bunch of skill ratings (pace, shooting, passing, dribbling, defending, physical, etc.) plus some physical info (height, weight, age, preferred foot...). The question I wanted to answer: can a model figure out what position someone plays just by looking at those numbers?

This is a multi-class classification problem. I trained 4 different models on the same data and compared how well each one does at telling positions apart.

## Dataset

- File: `FC26_20250921.csv` — a FIFA/FC player ratings export.
- Raw shape: 18,405 players × 110 columns.
- Each player originally has a specific position (CB, ST, CM, GK, CDM, RB, LB, CAM, LM, RM, RW, LW), which I collapsed into 4 broader groups so the classes are more balanced and easier to work with:

| Group | Made up of | Count |
|---|---|---|
| Midfielder | CDM, CM, CAM, RM, LM | 6,864 |
| Defender | CB, RB, LB | 6,116 |
| Forward | RW, LW, ST | 3,363 |
| Goalkeeper | GK | 2,062 |

Before training, I dropped a bunch of columns that would basically be "cheating" — things like the position-specific overall ratings (`st`, `cb`, `cam`...) since those already give away the position. I also dropped stuff like market value, club/nation admin info, and IDs/URLs, since none of that describes how a player actually plays.

> **Note:** By default the target is the grouped version (`position_group`, 4 classes). If you'd rather predict the original, more specific positions (CB, ST, CM, GK...) instead of the broader groups, just go to cell 38 and change:
> ```python
> TARGET = "position_group"
> ```
> to
> ```python
> TARGET = "primary_position"
> ```

## Pipeline

Here's the flow I followed in the notebook, start to finish:

1. **Import & load** — read the CSV, take a first look at shape and column info.
2. **Target creation** — extract each player's primary position, then map it into the 4 broader groups described above.
3. **Cleaning** — drop leakage/irrelevant columns, fill missing skill values (mostly goalkeeper-only players missing outfield stats, and vice versa) with 0, and encode `preferred_foot` as 0/1.
4. **Exploratory analysis** — distribution plots for position groups, histograms and boxplots for skill/physical attributes by position, a correlation heatmap to spot redundant features, and an outlier check using the IQR method.
5. **Train/test split** — 80/20 split, stratified by position group so class proportions stay consistent, with the target label-encoded.
6. **Scaling** — features standardized with `StandardScaler` (needed for Logistic Regression and SVM; Random Forest and XGBoost use the raw, unscaled features since tree-based models don't need it).
7. **Model training** — trained 4 classifiers side by side: Logistic Regression, Random Forest, SVM (RBF kernel), and XGBoost. Class imbalance (goalkeepers being a much smaller group) is handled with `class_weight="balanced"` or sample weights.
8. **Evaluation & comparison** — accuracy, precision, recall, and F1 (macro + weighted) for each model, plus classification reports, confusion matrices, and a bar chart comparing macro F1 across all 4.

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| SVM (RBF) | 0.9041 | 0.9099 | 0.9197 | **0.9141** |
| XGBoost | 0.9049 | 0.9147 | 0.9118 | 0.9132 |
| Random Forest | 0.9011 | 0.9093 | 0.9131 | 0.9111 |
| Logistic Regression | 0.8902 | 0.8966 | 0.9110 | 0.9020 |

SVM edges out XGBoost slightly on macro F1, though honestly all 3 non-linear models (SVM, XGBoost, Random Forest) land within a point of each other. Logistic Regression trails a bit behind but still does a solid job.

Goalkeepers are classified almost perfectly by every model (their stats are just wildly different from outfield players). The real confusion happens between Midfielder and Forward, since those roles share a lot of overlapping skills like passing and dribbling.

## Installation & Usage

```bash
git clone https://github.com/wtf-amnn/football-player-position-classifier.git
cd football-player-position-classifier
pip install -r requirements.txt
jupyter notebook FootballPlayerClassification.ipynb
```

Run the cells top to bottom. That's it — no extra setup needed.
