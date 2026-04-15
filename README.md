# SMS Spam Classification System

A complete end-to-end NLP pipeline to classify SMS messages as **spam** or **ham**.

## Setup

1. Install dependencies: 
2. Download NLTK data (auto-handled in code).

3. Place your `spam.csv` dataset inside the `data/` folder.

4. Run the project:
## Dataset
SMS Spam Collection Dataset from Kaggle.
Columns expected: `v1` (label), `v2` (message text).

## Output
- Model comparison table printed in terminal
- Confusion matrix plots saved in `outputs/`
- Best model saved in `models/`
