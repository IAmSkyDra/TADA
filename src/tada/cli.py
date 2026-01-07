import argparse
from tada.runner import run_training
from tada.logging_utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="TADA Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    data = parser.add_argument_group("Data Config")
    data.add_argument("--dataset_path", type=str, required=True, help="Path to TRAIN csv")
    data.add_argument("--test_path", type=str, default=None, help="Path to TEST csv (Optional)") # <--- MỚI THÊM
    data.add_argument("--source_col", type=str, default="bahnaric")
    data.add_argument("--target_col", type=str, default="vietnamese")
    data.add_argument("--max_source_len", type=int, default=128)
    data.add_argument("--max_target_len", type=int, default=128)

    # Augmentation
    aug = parser.add_argument_group("TADA Augmentation")
    aug.add_argument("--augment_method", type=str, default="baseline",
                     choices=["baseline", "combine", "swap", "theme", "synonym", 
                              "insertion", "deletion", "sliding", "delete_orig"],
                     help="Augmentation Strategy")
    
    aug.add_argument("--dictionary_path", type=str, default=None, 
                     help="Path to dictionary CSV (Required for theme, synonym, insertion)")
    
    aug.add_argument("--batch_size_aug", type=int, default=10, help="For Combine")
    aug.add_argument("--window_size", type=int, default=2, help="For Sliding")
    aug.add_argument("--num_deletions", type=int, default=1, help="For Deletion")

    # Training
    train = parser.add_argument_group("Training Config")
    train.add_argument("--model_name_or_path", type=str, default="vinai/bartpho-syllable")
    train.add_argument("--output_dir", type=str, default="outputs")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--fp16", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger()
    run_training(args)

if __name__ == "__main__":
    main()