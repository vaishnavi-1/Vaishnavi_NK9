import pandas as pd

# Read filtered dataset
df = pd.read_excel("english_hindi_final.xlsx")

# Randomly sample 200 rows
sample = df.sample(n=200, random_state=42)

# Save
sample.to_excel("sample_200.xlsx", index=False)

print("Done!")
