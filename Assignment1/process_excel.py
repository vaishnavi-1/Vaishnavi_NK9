
import pandas as pd

# Read Excel
df = pd.read_excel("english_hindi_filtered.xlsx")

# Word counts
df["English Word Count"] = df["English"].astype(str).apply(lambda x: len(x.split()))
df["Hindi Word Count"] = df["Hindi"].astype(str).apply(lambda x: len(x.split()))

# Difference in word count
df["Word Count Difference"] = (
    df["English Word Count"] - df["Hindi Word Count"]
)

# Character counts
df["English Character Count"] = df["English"].astype(str).apply(len)
df["Hindi Character Count"] = df["Hindi"].astype(str).apply(len)

# Difference in character count
df["Character Count Difference"] = (
    df["English Character Count"] -
    df["Hindi Character Count"]
)

# Apply filters
filtered = df[
    (df["English Word Count"].between(3, 60)) &
    (df["Hindi Word Count"].between(3, 60)) &
    (df["Word Count Difference"].between(-10, 10))
]

# Keep only first 10000 rows
filtered = filtered.head(10000)

# Save new Excel
filtered.to_excel("english_hindi_final.xlsx", index=False)

print("Done!")
print("Rows:", len(filtered))
