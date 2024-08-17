import pandas as pd
import numpy as np
from jiwer import wer, cer
import os
import csv

directory_path = "../corrections-sheet-tsvs"
csv_file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.tsv')]


# Load the CSV file
def load_data(file_path):
    original_sentences = []
    corrected_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='"')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 4:
                original_sentences.append(row[2])
                corrected_sentences.append(row[3])
    
    return original_sentences, corrected_sentences

# Calculate WER and CER for each sentence pair
def calculate_error_rates(original_sentences, corrected_sentences):
    wers = []
    cers = []
    changed = 0
    
    for i, (orig, corr) in enumerate(zip(original_sentences, corrected_sentences)):
        if orig != corr:
            changed += 1
        
        try:
            wers.append(wer(orig, corr))
            cers.append(cer(orig, corr))
        except ValueError as e:
            print(f"Error processing line {i+1}:")
            print(f"Original: {orig}")
            print(f"Corrected: {corr}")
            print(f"Error: {e}\n")
            continue  # Skip this pair and move to the next
    
    avg_wer = np.mean(wers) if wers else float('nan')
    avg_cer = np.mean(cers) if cers else float('nan')
    
    return avg_wer, avg_cer, changed, len(wers)

total_sentences = 0
weighted_wer_sum = 0
weighted_cer_sum = 0

for i, file_path in enumerate(csv_file_paths):
    original_sentences, corrected_sentences = load_data(file_path)
    avg_wer, avg_cer, changed, num_sentences = calculate_error_rates(original_sentences, corrected_sentences)
    total_sentences += num_sentences
    weighted_wer_sum += avg_wer * num_sentences
    weighted_cer_sum += avg_cer * num_sentences
    print(f'File {i+1}: {file_path}')
    print(f'  Number of Sentences: {num_sentences}')
    print(f'  Average WER: {avg_wer * 100:.2f}%')
    print(f'  Average CER: {avg_cer * 100:.2f}%\n')
    print(f'  Ratio Changed: {changed}/{num_sentences}\n')

# Calculate weighted averages for combined dataset
combined_wer = weighted_wer_sum / total_sentences
combined_cer = weighted_cer_sum / total_sentences

print('Combined Dataset:')
print(f'  Total Number of Sentences: {total_sentences}')
print(f'  Weighted Average WER: {combined_wer * 100:.2f}%')
print(f'  Weighted Average CER: {combined_cer * 100:.2f}%')