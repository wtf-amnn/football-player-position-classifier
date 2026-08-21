# Football Player Position Classifier

I built this to see if you can guess a football player's position just from their FIFA/FC stats — no name, no club, nothing, just numbers like pace, passing, defending, etc.

Turns out you can, pretty well.

## What it does

Takes player attribute data from FC26 and predicts which of 4 position groups a player belongs to:

- Goalkeeper
- Defender
- Midfielder
- Forward

The original dataset has more specific positions (CB, ST, CAM, etc.) but I grouped them into these 4 buckets to keep things balanced and simpler to classify.

## The data

Using `FC26_20250921.csv` (FIFA/FC player ratings export). Before training anything I dropped a bunch of columns that would be "cheating" — like the position-specific overall ratings (`st`, `cb`, `cam`...) since those basically already tell you the position. Also dropped stuff like market value, club info, and other things that don't actually describe how a player plays.

## Models used

I trained and compared 4 different classifiers:

1. **Logistic Regression**
2. **Random Forest**
3. **SVM (RBF kernel)**
4. **XGBoost**

Each one was trained on an 80/20 train-test split, with class balancing applied since goalkeepers are a much smaller group than the rest. Logistic Regression and SVM use scaled features, while Random Forest and XGBoost use the raw ones (they don't need scaling).

## How it's evaluated

For each model I looked at accuracy, precision, recall, and F1-score (both macro and weighted), plus confusion matrices to see exactly where each model gets confused — usually between Midfielder and Forward, or Midfielder and Defender, since those roles share a lot of overlapping skills.

## Running it

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then just open the notebook and run the cells top to bottom:

```bash
jupyter notebook FootballPlayerClassification.ipynb
```

Make sure `FC26_20250921.csv` is sitting in the same folder as the notebook before you run it.

## Notes

- Goalkeepers are basically the easiest to classify (their stats are wildly different from outfield players).
- The trickiest calls are usually in midfield, since a box-to-box or attacking mid can look statistically similar to a winger or striker.
- This was mostly a learning project to compare how different model types handle the same classification problem, not something meant for production use.
