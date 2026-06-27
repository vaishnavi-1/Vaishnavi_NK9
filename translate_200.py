from google import genai
import pandas as pd
import sacrebleu
import time

# ==========================
# Enter your Gemini API key
# ==========================
client = genai.Client(api_key="AQ.Ab8RN6KYX26niIPGEFkciQDKOdLU4k7MJ_tCEtgRNqFZozv_sw")

# ==========================
# Read Excel
# ==========================
df = pd.read_excel("sample_200.xlsx")

# Add columns if they don't exist
if "Generated Hindi" not in df.columns:
    df["Generated Hindi"] = ""

if "BLEU" not in df.columns:
    df["BLEU"] = 0.0

if "chrF" not in df.columns:
    df["chrF"] = 0.0

results = []

# ==========================
# Translate
# ==========================
for i, row in df.iterrows():

    # Skip already translated rows
    if pd.notna(row["Generated Hindi"]) and str(row["Generated Hindi"]).strip() != "":
        print(f"Skipping {i+1}")
        continue

    english = str(row["English"])
    reference = str(row["Hindi"])

    print(f"Translating {i+1}/{len(df)}")

    while True:

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"""
Translate the following English sentence into fluent natural Hindi.

Return ONLY the Hindi translation.

English:
{english}
"""
            )

            prediction = response.text.strip()

            break

        except Exception as e:

            print(e)
            print("Retrying in 20 seconds...")
            time.sleep(20)

    # -----------------------
    # Evaluation
    # -----------------------

    bleu = sacrebleu.sentence_bleu(
        prediction,
        [reference]
    ).score

    chrf = sacrebleu.sentence_chrf(
        prediction,
        [reference]
    ).score

    # Store in dataframe
    df.loc[i, "Generated Hindi"] = prediction
    df.loc[i, "BLEU"] = round(bleu, 2)
    df.loc[i, "chrF"] = round(chrf, 2)

    # Save after EVERY sentence
    df.to_excel("translated_200.xlsx", index=False)

    results.append({
        "English": english,
        "Reference": reference,
        "Prediction": prediction,
        "BLEU": round(bleu,2),
        "chrF": round(chrf,2)
    })

    time.sleep(3)

# ==========================
# Corpus Scores
# ==========================

predictions = df["Generated Hindi"].astype(str).tolist()
references = df["Hindi"].astype(str).tolist()

corpus_bleu = sacrebleu.corpus_bleu(
    predictions,
    [references]
).score

corpus_chrf = sacrebleu.corpus_chrf(
    predictions,
    [references]
).score

# ==========================
# Save TXT
# ==========================

with open("translation_results.txt","w",encoding="utf-8") as f:

    for _, row in df.iterrows():

        f.write("="*80 + "\n")

        f.write("English:\n")
        f.write(str(row["English"]) + "\n\n")

        f.write("Reference Hindi:\n")
        f.write(str(row["Hindi"]) + "\n\n")

        f.write("Generated Hindi:\n")
        f.write(str(row["Generated Hindi"]) + "\n\n")

        f.write(f"BLEU : {row['BLEU']}\n")
        f.write(f"chrF : {row['chrF']}\n\n")

    f.write("="*80 + "\n")
    f.write(f"Corpus BLEU : {corpus_bleu:.2f}\n")
    f.write(f"Corpus chrF : {corpus_chrf:.2f}\n")

print("\nFinished Successfully!")
print("Excel Saved : translated_200.xlsx")
print("TXT Saved   : translation_results.txt")
