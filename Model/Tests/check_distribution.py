import pandas as pd

# Load CSV and fix data
df = pd.read_csv('Model/Scores.csv', header=None, names=['Image', 'Score'])

# Drop rows where Score is not numeric
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df.dropna(subset=['Score'], inplace=True)
df['Score'] = df['Score'].astype(int)

# Count score frequencies
score_counts = df['Score'].value_counts().sort_index()

# Constants
MAX_BAR_WIDTH = 50
SCALE = MAX_BAR_WIDTH / score_counts.max()

# Header
print("Score Distribution")
print(f"Total Images: {len(df):,}\n")

# Visual Bars
for score in range(df['Score'].min(), df['Score'].max() + 1):
    count = score_counts.get(score, 0)
    bar = '*' * int(count * SCALE)
    percentage = (count / len(df)) * 100
    print(f"Score {score}: {bar.ljust(MAX_BAR_WIDTH)} {count:,} images ({percentage:.1f}%)")

# Statistics
print("\nStatistics:")
print(f"- Mean Score: {df['Score'].mean():.2f}")
print(f"- Score Range: {df['Score'].min()} to {df['Score'].max()}")
print(f"- Most Common: Score {score_counts.idxmax()} ({score_counts.max():,} images)")
