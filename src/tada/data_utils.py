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
    logger.info(f"Loading data from {args.dataset_path}")
    
    # 1. Load Train
    try:
        if args.dataset_path.endswith('.csv'):
            df_train = pd.read_csv(args.dataset_path)
        else:
            from datasets import load_dataset
            ds = load_dataset(args.dataset_path)
            df_train = ds['train'].to_pandas()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise e

    # Column Mapping
    if args.source_col not in df_train.columns:
        logger.warning(f"Column '{args.source_col}' missing. Using col 0 & 1.")
        args.source_col = df_train.columns[0]
        args.target_col = df_train.columns[1]

    # 2. Augmentation Logic
    # Nếu mode là 'train', ta bỏ qua augmentation (giả sử input đã là data xịn)
    if args.mode == "train":
        logger.info("Mode 'train': Skipping augmentation step.")
        final_df = df_train
    else:
        # Mode 'augment' hoặc 'all': Chạy augmentation
        methods = args.augment_method if isinstance(args.augment_method, list) else [args.augment_method]
        logger.info(f"Applying Methods: {methods}")

        # Danh sách chứa các DataFrame kết quả (Bắt đầu với Original)
        dfs_to_merge = [df_train]

        for method in methods:
            if method == "baseline": continue
            
            logger.info(f"Running generator: {method}...")
            aug = None
            
            if method == "combine":
                aug = Combine(args.source_col, args.target_col, df_train, batch_size=args.batch_size_aug)
            elif method == "swap":
                aug = SwapSentences(args.source_col, args.target_col, df_train)
            elif method == "sliding":
                aug = SlidingWindows(args.source_col, args.target_col, df_train, window_size=args.window_size)
            elif method == "deletion":
                aug = Deletion(args.source_col, args.target_col, df_train, num_deletions=args.num_deletions)
            elif method == "delete_orig":
                aug = DeletionWithOriginal(args.source_col, args.target_col, df_train, num_deletions=args.num_deletions)
            
            # Dictionary methods
            elif method in ["theme", "synonym", "insertion"]:
                if not args.dictionary_path:
                    raise ValueError(f"Method '{method}' requires --dictionary_path")
                
                if method == "theme":
                    aug = ReplaceWithSameThemes(args.source_col, args.target_col, df_train, args.dictionary_path)
                elif method == "synonym":
                    aug = ReplaceWithSameSynonyms(args.source_col, args.target_col, df_train, args.dictionary_path)
                elif method == "insertion":
                    aug = RandomInsertion(args.source_col, args.target_col, df_train, args.dictionary_path)
            
            if aug:
                res_df = aug.augment()
                dfs_to_merge.append(res_df)

        # Gộp tất cả kết quả lại
        final_df = pd.concat(dfs_to_merge)
        # Loại bỏ trùng lặp (nếu Original bị thêm nhiều lần)
        final_df = final_df.drop_duplicates(subset=[args.source_col, args.target_col])
        
        logger.info(f"Merged Size: {len(final_df)} rows")

        # NẾU CHỈ AUGMENT: Lưu file và Dừng
        if args.mode == "augment":
            if not args.save_data_path:
                raise ValueError("Mode 'augment' requires --save_data_path to save result.")
            
            final_df.to_csv(args.save_data_path, index=False)
            logger.info(f"Successfully saved augmented data to: {args.save_data_path}")
            return None  # Trả về None để báo hiệu runner dừng lại

    # 3. Load Test Data
    test_path = None
    if args.test_path:
        test_path = args.test_path
    else:
        data_dir = os.path.dirname(args.dataset_path)
        possible = os.path.join(data_dir, "test.csv")
        if os.path.exists(possible): test_path = possible

    if test_path and os.path.exists(test_path):
        logger.info(f"Loading test: {test_path}")
        df_test = pd.read_csv(test_path)
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(final_df),
            "test": Dataset.from_pandas(df_test)
        })
    else:
        logger.warning("No test file. Using random split.")
        full_ds = Dataset.from_pandas(final_df)
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