# TADA: Typology-Aware Data Augmentation for Low-Resource NMT
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ACL](https://img.shields.io/badge/ACL-Under%20Review-blue.svg)](#)

TADA is a principled framework for data augmentation in extremely low-resource Neural Machine Translation (NMT). It is explicitly grounded in linguistic typology and designed to support systematic, reproducible experimentation on Vietnamese ethnic minority languages.

> This repository accompanies the paper:  
> *When Data Augmentation Hurts: Typology-Aware and Meaning-Preserving Augmentation for Low-Resource Neural Machine Translation*  
> Under review for ACL Rolling Review – January 2026.

---

## The Heritage Behind the Data: A Tale of Two Cultures
![pannel](https://github.com/user-attachments/assets/4b380737-23cb-4973-ac81-279f8f860a96)

Behind the technical framework of Typology-Aware Data Augmentation (TADA) lies the cultural essence of millions. This project focuses on two ethnic groups in Vietnam with profound historical legacies that remain significantly underrepresented in the digital text landscape: the Tày and the Bahnar. Together, they reveal a striking contrast not only between demographic scale and digital presence, but also between fundamentally different linguistic typologies.

### The Tày: Echoes from the Northern Highlands

Residing in the emerald valleys of Northern Vietnam, the Tày people preserve a heritage as ancient as the mountains themselves. With a population of nearly two million people, the Tày constitute the second-largest ethnic group in Vietnam after the Kinh, accounting for almost 2% of the national population. Their language, belonging to the Tai–Kadai family, is the medium for *Then* singing—an art form recognized by UNESCO that serves as a spiritual bridge between communities.

* **Linguistic Essence:** The Tày language is a quintessential **Analytic** language. Its meaning is primarily constructed through precise word order and subtle particles rather than inflectional or derivational morphology.
* **The Digital Gap:** Despite their demographic prominence, the Tày language remains extremely scarce in the digital text landscape. This mismatch between population size and digital representation makes Tày a compelling case where low-resource NMT is not caused by marginality, but by structural neglect. TADA aims to amplify this presence by transforming limited written records into robust training data while respecting strict syntactic constraints.

### The Bahnar: Epics of the Central Highlands

In the heart of the Central Highlands, beneath the towering roofs of the Rong houses, live the Bahnar people. In contrast to the Tày, the Bahnar population is much smaller, comprising roughly 300,000 individuals, or about 0.3% of Vietnam’s population. Their culture is a symphony of gongs and oral epics (*H’môn*), many of which have been transcribed and preserved through generations.

* **Linguistic Essence:** In stark contrast to the analytic structure of Tày, Bahnar is a **Morphologically Rich** language of the Austroasiatic (Mon–Khmer) family. It employs a complex system of prefixes, infixes, and derivational processes to modify meaning—an internal word structure that traditional NMT models often struggle to capture.
* **The Mission:** As the Bahnar community navigates the digital era, their already limited textual footprint faces the risk of further underrepresentation. TADA seeks to support the preservation of this linguistic heritage by ensuring that NMT models can meaningfully process and learn from the intricate morphological structure of Bahnar words.

### Why Tày and Bahnar?

By focusing on **Tày (Analytic)** and **Bahnar (Morphologically Complex)**, TADA does not merely address a technical challenge in low-resource NMT. Instead, it deliberately spans two extremes:

- a large but digitally underrepresented ethnic group, and  
- a small and structurally complex language community.

This pairing allows us to study how data augmentation interacts with typology under vastly different demographic and linguistic conditions. We believe that NMT for low-resource languages is not merely a matter of increasing data volume—it is about respecting linguistic structure, honoring cultural identity, and ensuring the survival of written mother tongues in the age of Artificial Intelligence.

---

## Why TADA?

TADA is motivated by the observation that augmentation in low-resource NMT is often applied in a typology-agnostic way, which can lead to unstable behavior or even degraded translation quality. To address this, TADA organizes augmentation and training around typological constraints and meaning preservation, so that improvements are both measurable and interpretable.

**Typology-Aware.** TADA moves beyond one-size-fits-all augmentation by explicitly respecting linguistic constraints, distinguishing between analytic and morphologically complex language structures.

**Meaning-Preserving.**  TADA prioritizes semantic fidelity over blind structural variation, reducing semantic drift and error amplification in extremely low-resource settings.

**Full Pipeline.**  TADA integrates language-proximal initialization, monolingual continual pretraining, and supervised fine-tuning into a unified experimental framework.

---

## Core Features

TADA provides a comprehensive suite of components designed to support rigorous and interpretable evaluation of data augmentation strategies across typologically diverse languages. In practice, this means you can (i) generate augmented data under clearly defined operators, (ii) train under a consistent three-stage pipeline, and (iii) analyze outcomes with typology-aware diagnostics.

### Augmentation Operators

- **Meaning-Preserving Augmentations:**  Synonym Replacement, Theme-Based Replacement, Contextual Place/Time Insertion.

- **Structure-Altering Augmentations (Robustness Tests):**  Thematic Concatenation, Sentence Reordering, Sliding Window Segmentation.

- **Deletion Strategies:**  Exhaustive Deletion and Deletion + Original (composite augmentation).

### Three-Stage Training Pipeline

- **Stage 1: Pretrained Initialization.**  Initializes models from BARTPho (Vietnamese) to exploit typological and lexical proximity.

- **Stage 2: Continual Language Model Pretraining.**  Adapts the model to minority languages using low-resource monolingual corpora.

- **Stage 3: Supervised Fine-tuning.**  Trains on TADA-augmented parallel data with controlled and reproducible experimental settings.

### Linguistic Analysis and Evaluation

- Provides fine-grained analysis explaining why certain augmentation strategies succeed for analytic languages (Tày) but fail for morphologically rich languages (Bahnar).
- Supports standard machine translation evaluation metrics, including BLEU and METEOR.
- Releases benchmark datasets and results for Tày–Vietnamese and Bahnar–Vietnamese translation.
- **Genealogy-Driven Design:** Explicitly distinguishes analytic constraints (Tai–Kadai) from morphological constraints (Austroasiatic) to anticipate augmentation stability and failure modes.

---

## Command-Line Interface (CLI)

TADA is designed as a command-line tool, `tada-train`. The CLI exposes a small set of arguments that cover data loading, augmentation configuration, and training hyperparameters, making it straightforward to reproduce paper-style experiments.

```text
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

Getting started with TADA requires setting up the environment and installing dependencies for BART-based training. If you already have a working PyTorch + CUDA setup, the steps below should be enough to install the package in editable mode for development and experimentation.

```bash
git clone https://anonymous.4open.science/r/TADA.git
cd TADA
pip install -e .
```

---

## Supported Augmentation Methods

Below is the list of available augmentation strategies you can pass to `--augment_method`. Each method is implemented as a controlled operator so that comparisons are consistent across languages and experimental runs.

| Method Name | Description | Related Arguments |
| :--- | :--- | :--- |
| **`baseline`** | Standard fine-tuning on original data (No augmentation). | (None) |
| **`combine`** | Concatenates N consecutive sentences into a single training sample. | `--batch_size_aug` (Default: 10) |
| **`sliding`** | Creates a sliding window of $N$ sentences. | `--window_size` (Default: 2) |
| **`deletion`** | Randomly deletes $K$ words from the source sentence. | `--num_deletions` (Default: 1) |
| **`delete_orig`** | Keeps the **Original** sentence AND adds the **Deletion** version. | `--num_deletions` (Default: 1) |
| **`swap`** | Swaps the order of sentences or phrases. | (None) |
| **`synonym`** | Replaces words with their synonyms found in the dictionary. | `--dictionary_path` **(Required)** |
| **`theme`** | Replaces words with others from the same semantic theme (e.g., *river* -> *stream*). | `--dictionary_path` **(Required)** |
| **`insertion`** | Randomly inserts related words from the dictionary into the sentence. | `--dictionary_path` **(Required)** |

---

## Quickstart

This section demonstrates how to run experiments using the `tada-train` command. The examples below are written to be readable and copy-pastable, while keeping the configuration explicit so that results can be reproduced reliably.

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

**2. Train Only (`--mode train`)**

This mode trains on the provided data without generating new augmented samples. In our experiments, it is often used to evaluate training stability under controlled conditions.

```bash
tada-train --mode train \
           --dataset_path "data/Bahnar/Original/train.csv" \
           --epochs 10 \
           --output_dir "outputs/Bahnar_TrainOnly"
```

**3. Augment + Train Pipeline (`--mode all`)**

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

Our benchmark encompasses two typologically distinct low-resource languages paired with Vietnamese. These datasets were constructed through fieldwork and community collaboration, and they are intended to support controlled comparisons across typological conditions.

**Tày–Vietnamese (Tai–Kadai)**  
- Structure: Analytic, fixed SVO order, tonal.  
- Source: Fieldwork, textbooks, and dictionaries from northern Vietnam.  
- Size: ~20,554 sentence pairs.  
- Characteristics: Tolerates surface-level token edits (swapping, deletion).

**Bahnar–Vietnamese (Austroasiatic)**  
- Structure: Agglutinative, rich morphology (prefixation/infixation), non-tonal.  
- Source: Elicitation sessions, religious books, folksongs from Central Highlands.  
- Size: ~51,930 sentence pairs.  
- Characteristics: Highly sensitive to structural perturbation; requires morphology-aware augmentation.

Detailed statistics and collection procedures are available in Appendix B of the paper.

---

## Data Format

TADA expects input data in **CSV format**. To make runs consistent across environments, we recommend following the directory layout below, which mirrors the default paths used in the example commands.

**Directory Structure:**  Ensure your data directory is organized as follows:

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
│    │       ├── combine.csv
│    │       └── delete_original.csv
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

## File Content (Line-aligned)

The following snippets illustrate the expected column structure for the core files. In practice, you can add more rows as needed as long as the header and column ordering remain consistent with your CLI arguments.

**1. `train.csv`**

```text
Bahnaric,Vietnamese
"pơm.","thực hiện"
```

**2. `dictionary.csv`**

```text
Bahnaric,Vietnamese,pos,theme
"pơlĕi pơla","bản làng",d.,place
```

---

## Outputs

Upon execution, TADA creates an `outputs/` directory organized by experiment run. This makes it easier to track checkpoints, training states, and evaluation artifacts across multiple augmentation settings.

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

## Results 

To facilitate analysis across typologically different languages, TADA supports comparative reporting and consolidated experiment tracking. The summary below highlights the core empirical trends emphasized by the framework.

**Key Findings:**

- **Typology Matters:** Strategies successful for Tày (Analytic) often degrade performance for Bahnar (Agglutinative).
- **Best Performer:** Deletion + Original consistently yields the strongest improvements across both languages (e.g., +14.7 BLEU on Tày–Vi).
- **Failure Modes:** Insert + Swap degrades Bahnar performance significantly due to morphological violation.

**Example Summary (BLEU Scores):**

<img width="1038" height="610" alt="Results and Analysis" src="https://github.com/user-attachments/assets/f902e9a5-887f-4b1a-94c4-c3d996027b2d" />

### Code Structure

The project follows a modular Python package structure, with a clear separation between CLI logic, runners, augmentation operators, and evaluation utilities:

```bash
src/tada/
├── cli.py
├── runner.py
├── augmentation.py
├── data_utils.py
├── metrics.py
└── logging_utils.py
```

TADA is designed for strict reproducibility in extremely low-resource settings.

### Reproducibility

TADA emphasizes reproducibility for low-resource research by standardizing common sources of variance:

- **Seeds:** Fixed random seeds (default: 42) for all data splits and initialization.  
- **Model:** Built on `vinai/bartpho` [1] to ensure a consistent pretrained baseline.  
- **Splits:** Deterministic train/test splitting logic.

---

## Citation

If you use TADA in your research, please cite our work:

```bibtex
@inproceedings{anonymous2025tada,
    title = {When Data Augmentation Hurts: Typology-Aware and Meaning-Preserving Augmentation for Low-Resource Neural Machine Translation},
    author = {Anonymous},
    booktitle = {ACL Submission},
    year = {2026}
}
```

---

## Ethics Statement

This project is conducted with a strong commitment to ethical research practices
and respect for the language communities involved. All datasets were collected
through fieldwork and community collaboration, drawing on publicly available
educational materials, dictionaries, and culturally shared texts. No personal,
private, or sensitive information is included at any stage of data collection
or processing.

The primary goal of TADA is to support language preservation and responsible
development of NLP technologies for underrepresented and low-resource languages.
We emphasize that the framework is intended for research and educational purposes,
and we encourage users to engage with these linguistic resources in a manner that
respects cultural context, community ownership, and long-term sustainability.

---

## License

This project is released under the **MIT License**, a permissive open-source
license that allows reuse, modification, and distribution with minimal
restrictions. The license is intended to facilitate broad adoption of TADA
in both academic research and practical NLP applications, while preserving
appropriate attribution to the original authors.

---

## References

[1] Nguyen L. Tran, D. M. Le, and D. Q. Nguyen. "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese". Interspeech, 2022.
