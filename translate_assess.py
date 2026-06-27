import os
import pandas as pd
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import sacrebleu

# 1. Load the Model and Tokenizer (Runs completely locally without API)
print("Loading M2M100 translation model... (This might take a couple of minutes on first run)")
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Set source language to English
tokenizer.src_lang = "en"

# Move model to GPU if available for faster execution
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 2. Load your 200 English sentences
# TODO: Replace this placeholder list with your actual 200 sentences from Assignment 1
# Example: If you have an existing Excel file, use: df = pd.read_excel("assignment1.xlsx")
english_sentences = [
    "Artificial intelligence is transforming the world.",
    "Data science requires a strong understanding of mathematics.",
    "Natural language processing helps computers understand human language.",
    # ... Add all 200 sentences here ...
]

# Ensure you have exactly 200 sentences or slice them
english_sentences = english_sentences[:200]

# Optional: If you have reference Hindi translations to compute BLEU/CHRF/TER
# Replace these with actual reference Hindi sentences from Assignment 1 if available.
# If you don't have reference translations, these metrics will evaluate against placeholder text.
reference_hindi_sentences = [
    "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
    "डेटा विज्ञान के लिए गणित की मजबूत समझ की आवश्यकता होती है।",
    "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में मदद करता है।",
]

# 3. Perform Local Translation Loop
print(f"Translating {len(english_sentences)} sentences using {device}...")
translated_hindi_sentences = []

for idx, sentence in enumerate(english_sentences):
    # Tokenize input text
    encoded_en = tokenizer(sentence, return_tensors="pt").to(device)
    
    # Generate translation forcing Hindi ("hi") as the target language
    generated_tokens = model.generate(
        **encoded_en,
        forced_bos_token_id=tokenizer.get_lang_id("hi")
    )
    
    # Decode tokens back to a readable string
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_hindi_sentences.append(translation)
    
    if (idx + 1) % 10 == 0:
        print(f"Progress: {idx + 1}/{len(english_sentences)} sentences translated.")

# 4. Save to Excel format as requested
df_output = pd.DataFrame({
    'Column A': english_sentences,
    'Column B': translated_hindi_sentences
})
output_excel_path = "Translation_Output.xlsx"
df_output.to_excel(output_excel_path, index=False)
print(f"Saved translations to {output_excel_path}")

# 5. Calculate Evaluation Scores (BLEU, CHRF, TER)
print("Calculating evaluation metrics...")

# Note: Metrics require ground-truth references to make sense.
# We adjust the reference size to match our translations for calculation demonstration.
refs = [reference_hindi_sentences[:len(translated_hindi_sentences)]]
sys = translated_hindi_sentences

# Compute metrics
bleu = sacrebleu.corpus_bleu(sys, refs)
chrf = sacrebleu.corpus_chrf(sys, refs)
ter = sacrebleu.corpus_ter(sys, refs)

# Save scores to a .txt file
metrics_output_path = "evaluation_scores.txt"
with open(metrics_output_path, "w", encoding="utf-8") as f:
    f.write("=== Translation Evaluation Metrics ===\n")
    f.write(f"BLEU Score: {bleu.score:.4f}\n")
    f.write(f"CHRF Score: {chrf.score:.4f}\n")
    f.write(f"TER Score:  {ter.score:.4f}\n")

print(f"Saved evaluation scores to {metrics_output_path}")
print("Process completed successfully!")
