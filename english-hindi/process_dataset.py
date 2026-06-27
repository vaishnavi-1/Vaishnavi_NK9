import pandas as pd

# Read English file
with open("eng.txt", "r", encoding="utf-8") as f:
    eng = [line.strip() for line in f]

# Read Hindi file
with open("hin.txt", "r", encoding="utf-8") as f:
    hin = [line.strip() for line in f]

# Ensure equal number of lines
length = min(len(eng), len(hin))

eng = eng[:length]
hin = hin[:length]

# Create DataFrame
df = pd.DataFrame({
    "English": eng,
    "Hindi": hin
})

# Count words
df["English Word Count"] = df["English"].apply(lambda x: len(x.split()))
df["Hindi Word Count"] = df["Hindi"].apply(lambda x: len(x.split()))

# Keep sentences with 3–60 words in BOTH languages
filtered = df[
    (df["English Word Count"] >= 3) &
    (df["English Word Count"] <= 60) &
    (df["Hindi Word Count"] >= 3) &
    (df["Hindi Word Count"] <= 60)
]

# Keep first 10,000 rows
filtered = filtered.head(10000)

# Save CSV
filtered.to_csv(
    "english_hindi_filtered.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save Excel
filtered.to_excel(
    "english_hindi_filtered.xlsx",
    index=False
)

print("Done!")
print("Rows:", len(filtered))
