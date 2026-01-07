# TADA: Typology-Aware Data Augmentation for Low-Resource NMT
A principled, typology-aware framework for data augmentation in extremely low-resource Neural Machine Translation (NMT), specifically designed for Vietnamese ethnic minority languages.

This repository accompanies the paper: When Data Augmentation Hurts: Typology-Aware and Meaning-Preserving Augmentation for Low-Resource Neural Machine Translation Anonymous ACL Submission.
---

## The Heritage Behind the Data: A Tale of Two Cultures
![pannel](https://github.com/user-attachments/assets/503f852f-c957-4a46-a8eb-297bd110815d)

Behind the technical framework of TADA lies the cultural essence of millions. This project focuses on two ethnic groups with profound historical legacies in Vietnam that remain significantly underrepresented in the digital text landscape: the Tay and the Bahnar.

### The Tày: Echoes from the Northern Highlands
Residing in the emerald valleys of Northern Vietnam, the Tày people preserve a heritage as ancient as the mountains themselves. Their language, belonging to the Tai-Kadai family, is the medium for "Then" singing—a UNESCO-recognized art form that serves as a spiritual bridge between communities.

* **Linguistic Essence:** The Tày language is a quintessential **Analytic** language. Its meaning is primarily constructed through precise word order and subtle particles rather than word transformations.
* **The Digital Gap:** Despite being the second-largest ethnic group in Vietnam, their textual presence in the digital world is a mere whisper. TADA aims to amplify this presence, transforming scarce written records into robust data for NMT by respecting its unique syntactic constraints.

### The Bahnar: Epics of the Central Highlands
In the heart of the Central Highlands, beneath the towering roofs of the Rong houses, live the Bahnar people. Their culture is a symphony of Gongs and oral epics (H'mon) that have been transcribed and passed down through generations.

* **Linguistic Essence:** In stark contrast to the tonal languages of the plains, Bahnar is a **Morphologically Rich** language of the Austroasiatic family. It employs a complex system of prefixes and infixes to modify meanings—a structural puzzle that traditional NMT models often fail to decode.
* **The Mission:** As the Bahnar community navigates the digital era, their unique written identity faces the risk of "digital extinction." TADA seeks to preserve this heritage by ensuring that NMT models can comprehend the intricate internal structures of Bahnar words.

### Why Tay and Bahnar?
By focusing on the **Tày (Analytic)** and the **Bahnar (Agglutinative/Morphologically Complex)**, TADA does not merely solve a technical challenge. We are building a bridge between two fundamentally different linguistic worlds. We believe that NMT for low-resource languages is not just a matter of data points—it is about honoring the dignity and ensuring the survival of a people's written mother tongue in the age of Artificial Intelligence.

---

## Why TADA?
Typology-Aware: Moves beyond "one-size-fits-all" augmentation by respecting linguistic constraints (Analytic vs. Agglutinative).

Meaning-Preserving: Prioritizes semantic fidelity over blind structural variation to prevent drift in low-resource settings.

Full Pipeline: Integrates language-proximal initialization, monolingual continual pretraining, and supervised fine-tuning.

---

## Core Features
TADA provides a comprehensive suite of features designed to support rigorous evaluation of augmentation strategies on typologically diverse languages.

**Augmentation Operators** 

Meaning-Preserving: Synonym Replacement, Theme-Based Replacement, Contextual Place/Time Insertion.

Structure-Altering (Robustness Tests): Thematic Concatenation, Sentence Reordering, Sliding Window Segmentation.

Deletion Strategies: Exhaustive Deletion and Deletion + Original (composite).

**Three-Stage Training Pipeline**

- Stage 1: Pretrained Initialization. Leverages BARTPho (Vietnamese) to utilize typological proximity.

- Stage 2: Continual LM Pretraining. Adapts the model to minority languages using low-resource monolingual corpora.

- Stage 3: Supervised Fine-tuning. Trains on TADA-augmented parallel data with controlled experimental settings.

**Linguistic Analysis & Evaluation**

Provides detailed analysis of why specific augmentations fail for morphologically rich languages (Bahnar) while working for analytic ones (Tày).

Supports BLEU and METEOR evaluation metrics.

Releases benchmarks for Tày-Vietnamese and Bahnar-Vietnamese translation.

Genealogy-Driven Design. Distinguishes between analytic constraints (Tai-Kadai) and agglutinative/morphological constraints (Austroasiatic) to predict augmentation stability.

---

## Command-Line Interface (CLI)
TADA is designed as a command-line tool tada-train. Below are the available options:

```
usage: tada-train [-h] --dataset_path DATASET_PATH [--test_path TEST_PATH] [--mode {augment,train,all}] ...

Train and evaluate typology-aware augmentation for Low-Resource NMT.

options:
  -h, --help            Show this help message and exit.

Data Arguments:
  --mode STR:           Mode: 'augment' (gen data only), 'train' (train only), 'all' (augment + train) (default: all).
  --save_data_path PATH Path to save augmented CSV (Required if mode='augment').
  --dataset_path PATH   Path to the input CSV file (Train set) (Required).
  --test_path PATH      Path to the input CSV file (Test set) (Optional).
  --source_col STR      Column name for source language (default: src).
  --target_col STR      Column name for target language (default: tgt).
  --max_source_len INT  Max sequence length for source text (default: 128).
  --max_target_len INT  Max sequence length for target text (default: 128).

Augmentation Arguments:
  --augment_method STR  Method: {baseline, combine, swap, theme, synonym, 
                        insertion, sliding, deletion, delete_orig} (default: baseline).
  --dictionary_path PATH Path to dictionary CSV (Required for theme/synonym/insertion).
  --batch_size_aug INT  Batch size for 'combine' method (default: 10).
  --window_size INT     Window size for 'sliding' method (default: 2).
  --num_deletions INT   Number of tokens to delete (default: 1).

Training Arguments:
  --model_name_or_path STR  Pretrained model (default: vinai/bartpho-syllable).
  --output_dir STR      Directory to save checkpoints (default: outputs).
  --epochs INT          Number of training epochs (default: 10).
  --batch_size INT      Batch size for training/eval (default: 16).
  --lr FLOAT            Learning rate (default: 2e-5).
  --seed INT            Random seed for reproducibility (default: 42).
  --fp16                Enable mixed precision training (FP16).
  
```
---
## Installation
Getting started with TADA requires setting up the environment and installing dependencies for BART-based training.


```bash

git clone https://anonymous.4open.science/r/TADA.git
cd TADA
pip install -e .

```

---
## Supported Augmentation Methods

Below is the list of available augmentation strategies you can pass to `--augment_method`.

| Method Name | Description | Related Arguments |
| :--- | :--- | :--- |
| **`baseline`** | Standard fine-tuning on original data (No augmentation). | (None) |
| **`combine`** | Concatenates N consecutive sentences into a single training sample. | `--batch_size_aug` (Default: 10) |
| **`sliding`** | Creates a sliding window of $N$ sentences. | `--window_size` (Default: 2) |
| **`deletion`** | Randomly deletes $K$ words from the source sentence. | `--num_deletions` (Default: 1) |
| **`delete_orig`** | Keeps the **Original** sentence AND adds the **Deletion** version. | `--num_deletions` (Default: 1) |
| **`swap`** | Swaps the order of sentences or phrases. | *(None)* |
| **`synonym`** | Replaces words with their synonyms found in the dictionary. | `--dictionary_path` **(Required)** |
| **`theme`** | Replaces words with others from the same semantic theme (e.g., *river* -> *stream*). | `--dictionary_path` **(Required)** |
| **`insertion`** | Randomly inserts related words from the dictionary into the sentence. | `--dictionary_path` **(Required)** |

---
## Quickstart
This section demonstrates how to run experiments using the tada-train command.
**1. Generate Augmented Data Only (`--mode augment`)**
*Example A: Single Method (Deletion)*
```bash
tada-train --mode augment \
           --dataset_path "data/Bahnar/Original/train.csv" \
           --augment_method deletion \
           --num_deletions 1 \
           --save_data_path "data/Bahnar/train_aug_delete.csv"
```
*Example B: Multiple Methods (Deletion + Synonym)*
```bash
tada-train --mode augment \
           --dataset_path "data/Bahnar/Original/train.csv" \
           --augment_method deletion synonym \
           --dictionary_path "data/Bahnar/Original/dictionary.csv" \
           --num_deletions 1 \
           --save_data_path "data/Bahnar/train_aug_multi.csv"
```
**2. Train only ('--mode train)**
This method combines the original corpus with deletion-augmented samples, often yielding the best stability for analytic languages.
```bash
tada-train --mode train \
           --dataset_path "data/Bahnar/Original/train.csv" \
           --epochs 10 \
           --output_dir "outputs/Bahnar_TrainOnly"
```

**3. Augment + Train Pipeline (--mode all)**
```bash
tada-train --mode all \
           --dataset_path "data/Bahnar/Original/train.csv" \
           --augment_method delete_orig combine \
           --batch_size_aug 5 \
           --epochs 15 \
           --output_dir "outputs/Bahnar_FullPipeline"
```
---

## Datasets
Our benchmark encompasses two typologically distinct low-resource languages paired with Vietnamese. These datasets were constructed through fieldwork and community collaboration.

**Tày-Vietnamese (Tai-Kadai)**

Structure: Analytic, fixed SVO order, tonal.

Source: Fieldwork, textbooks, and dictionaries from northern Vietnam.

Size: ~20,554 sentence pairs.

Characteristics: Tolerates surface-level token edits (swapping, deletion).

**Bahnar-Vietnamese (Austroasiatic)**

Structure: Agglutinative, rich morphology (prefixation/infixation), non-tonal.

Source: Elicitation sessions, religious books, folksongs from Central Highlands.

Size: ~51,930 sentence pairs.

Characteristics: Highly sensitive to structural perturbation; requires morphology-aware augmentation.

Detailed statistics and collection procedures are available in Appendix B of the paper.

---

## Data Format
TADA expects input data in **CSV format**.

**Directory Structure:**
Ensure your data directory is organized as follows:

```bash

TADA/
├── data/
│    ├── Bahnar/                    
│    │   ├── Original/              
│    │   │   ├── train.csv        
│    │   │   ├── test.csv           
│    │   │   └── dictionary.csv     
│    │   ├── Augmented (1-side)/   
│    │   │   ├── combine_bahnar.csv        
│    │   │   └── combine_vietnamese.csv
│    │   └── Augmented (2-side)/ 
│    │   │   ├── combine.csv        
│    │   │   └── delete_original.csv   
│    └── Tay/                       
│        ├── Original/
│        │   ├── train.csv
│        │   ├── test.csv
│        │   └── dictionary.csv
│        ├── Augmented (1-side)/
│        │   ├── combine_tay.csv        
│        │   └── combine_vietnamese.csv
│        └── Augmented (2-side)/    
│            ├── combine.csv        
│            └── delete_original.csv

``` 

---

# File Content (Line-aligned):
**1. train.csv**
```bash
Bahnaric,Vietnamese
"pơm.","thực hiện"
```
**2. dictionary.csv**
```bash
Bahnaric,Vietnamese,pos,theme
"pơlĕi pơla","bản làng",d.,place
```

---

## Outputs
Upon execution, TADA creates an outputs/ directory organized by experiment run.
```bash
outputs/
├── baseline_ep10_sd42/         
│   ├── checkpoint-500/        
│   ├── pytorch_model.bin       
│   ├── trainer_state.json      
│   └── all_results.json        
└── delete_orig_ep15_sd42/
    └── ...
```

---

## Benchmarks & Results
To facilitate analysis of results across typologically different languages, TADA allows for comparative reporting.

**Key Findings:**

**Typology Matters:** Strategies successful for Tày (Analytic) often degrade performance for Bahnar (Agglutinative).

**Best Performer:** Deletion + Original consistently yields the strongest improvements across both languages (e.g., +14.7 BLEU on Tày-Vi).

**Failure Modes:** Insert + Swap degrades Bahnar performance significantly due to morphological violation.

**Example Summary (BLEU Scores):**

<img width="940" height="551" alt="image" src="https://github.com/user-attachments/assets/b66a2155-f0c3-47c8-a304-b6c8bbe0623a" />


**Code Structure**
The project follows a modular Python package structure:

```bash
src/tada/
├── cli.py           
├── runner.py           
├── augmentation.py     
├── data_utils.py       
├── metrics.py          
└── logging_utils.py    
TADA is designed for strict reproducibility in extremely low-resource settings.
```
**Reproducibility**
TADA ensures strict reproducibility for low-resource research:

**Seeds:** Fixed random seeds (default: 42) for all data splits and initialization.

**Model:** Built on vinai/bartpho to ensure a consistent pre-trained baseline.

**Splits:** Deterministic train/test splitting logic.

---

## Citation
If you use TADA or the datasets in your research, please cite our work:

```
@inproceedings{anonymous2025tada,
    title = {When Data Augmentation Hurts: Typology-Aware and Meaning-Preserving Augmentation for Low-Resource Neural Machine Translation},
    author = {Anonymous},
    booktitle = {ACL Submission},
    year = {2026}
}
```
---

## References

[1] Nguyen L. Tran, D. M. Le, and D. Q. Nguyen. "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese". Interspeech, 2022.

[2] M. J. Alves. "Morphology in Austroasiatic Languages". 2019.

[3] B. Li, Y. Hou, and W. Che. "Data augmentation approaches in natural language processing: A survey". AI Open, 2022.
