import os
import re
import pandas as pd
import logging
from datasets import Dataset, DatasetDict
from tada.augmentation import (
    Combine, SlidingWindows, Deletion, DeletionWithOriginal,
    SwapSentences, ReplaceWithSameThemes, ReplaceWithSameSynonyms, RandomInsertion
)

logger = logging.getLogger(__name__)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\\n|\|\\|[\n\r\t]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_and_process_data(args, tokenizer):
    logger.info(f"Loading train data from {args.dataset_path}")
    
    # 1. Load Train
    try:
        if args.dataset_path.endswith('.csv'):
            df_train = pd.read_csv(args.dataset_path)
        else:
            from datasets import load_dataset
            ds = load_dataset(args.dataset_path)
            df_train = ds['train'].to_pandas()
    except Exception as e:
        logger.error(f"Failed to load train dataset: {e}")
        raise e

    # Column Mapping
    if args.source_col not in df_train.columns:
        logger.warning(f"Column '{args.source_col}' missing. Using col 0 & 1.")
        args.source_col = df_train.columns[0]
        args.target_col = df_train.columns[1]

    # 2. Augmentation (Only Train)
    method = args.augment_method
    logger.info(f"Applying Augmentation: {method}")

    if method == "baseline":
        aug_df = df_train
    elif method == "combine":
        aug = Combine(args.source_col, args.target_col, df_train, batch_size=args.batch_size_aug)
        aug_df = aug.augment()
    elif method == "swap":
        aug = SwapSentences(args.source_col, args.target_col, df_train)
        aug_df = aug.augment()
    elif method == "sliding":
        aug = SlidingWindows(args.source_col, args.target_col, df_train, window_size=args.window_size)
        aug_df = aug.augment()
    elif method == "deletion":
        aug = Deletion(args.source_col, args.target_col, df_train, num_deletions=args.num_deletions)
        aug_df = aug.augment()
    elif method == "delete_orig":
        aug = DeletionWithOriginal(args.source_col, args.target_col, df_train, num_deletions=args.num_deletions)
        aug_df = aug.augment()
    elif method in ["theme", "synonym", "insertion"]:
        if not args.dictionary_path:
            raise ValueError(f"Method '{method}' requires --dictionary_path")
        if method == "theme":
            aug = ReplaceWithSameThemes(args.source_col, args.target_col, df_train, args.dictionary_path)
        elif method == "synonym":
            aug = ReplaceWithSameSynonyms(args.source_col, args.target_col, df_train, args.dictionary_path)
        elif method == "insertion":
            aug = RandomInsertion(args.source_col, args.target_col, df_train, args.dictionary_path)
        aug_df = aug.augment()
    else:
        aug_df = df_train

    logger.info(f"Train Size: {len(df_train)} -> {len(aug_df)}")

    # 3. Load Test (Logic MỚI)
    test_path = None
    
    # Ưu tiên lấy từ tham số --test_path
    if args.test_path:
        test_path = args.test_path
    else:
        # Nếu không truyền, tự tìm file "test.csv" cùng thư mục với train
        data_dir = os.path.dirname(args.dataset_path)
        possible_path = os.path.join(data_dir, "test.csv")
        if os.path.exists(possible_path):
            test_path = possible_path

    if test_path and os.path.exists(test_path):
        logger.info(f"Loading test data from: {test_path}")
        df_test = pd.read_csv(test_path)
        
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(aug_df),
            "test": Dataset.from_pandas(df_test)
        })
    else:
        logger.warning("No test file found. Using random split (10%).")
        full_ds = Dataset.from_pandas(aug_df)
        dataset_dict = full_ds.train_test_split(test_size=0.1, seed=args.seed)

    # 4. Tokenize
    def preprocess(examples):
        inputs = [clean_text(str(x)) for x in examples[args.source_col]]
        targets = [clean_text(str(x)) for x in examples[args.target_col]]
        
        model_inputs = tokenizer(inputs, max_length=args.max_source_len, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_len, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset_dict.map(preprocess, batched=True)