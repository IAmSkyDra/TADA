import pandas as pd
import numpy as np
import random
import re
import logging
from itertools import permutations

logger = logging.getLogger(__name__)

class AugmentMethod:
    """Base class for augmentation strategies."""
    def __init__(self, source_col, target_col, data=None):
        self.source_col = source_col
        self.target_col = target_col
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("Data must be a pandas DataFrame")

    def augment(self):
        """Override this method in subclasses."""
        return self.data

class Combine(AugmentMethod):
    def __init__(self, source_col, target_col, data, batch_size=10):
        super().__init__(source_col, target_col, data)
        self.batch_size = batch_size
    
    def augment(self):
        logger.info(f"Applying Combine augmentation (batch_size={self.batch_size})...")
        data_vals = self.data.values
        combined_data = []
        
        for i in range(0, len(data_vals), self.batch_size):
            batch = data_vals[i : i + self.batch_size]
            if len(batch) < 2:
                pass 
            else:
                for a, b in permutations(batch, 2):
                    new_src = f"{a[0]} {b[0]}"
                    new_tgt = f"{a[1]} {b[1]}"
                    combined_data.append([new_src, new_tgt])
        
        return pd.DataFrame(combined_data, columns=[self.source_col, self.target_col])

class SwapSentences(AugmentMethod):
    def augment(self):
        logger.info("Applying SwapSentences...")
        data_vals = self.data.values
        swapped_data = []
        delimiters = ".;?!"
        
        for src, tgt in data_vals:
            src = str(src)
            tgt = str(tgt)
            
            sentences_a = [s.strip() for s in re.split(f'[{delimiters}]', src) if s]
            sentences_b = [s.strip() for s in re.split(f'[{delimiters}]', tgt) if s]
            
            if len(sentences_a) == len(sentences_b) and len(sentences_a) > 1:
                for perm in permutations(range(len(sentences_a))):
                   
                    perm_a = [sentences_a[i] for i in perm]
                    perm_b = [sentences_b[i] for i in perm]
                    
                    new_src = '. '.join(perm_a) + '.'
                    new_tgt = '. '.join(perm_b) + '.'
                    swapped_data.append([new_src, new_tgt])
            else:

                pass

        return pd.DataFrame(swapped_data, columns=[self.source_col, self.target_col])

class ReplaceWithSameThemes(AugmentMethod):
    def __init__(self, source_col, target_col, data, dictionary_path):
        super().__init__(source_col, target_col, data)
        self.dictionary_path = dictionary_path
        
        try:
            self.df_theme = pd.read_csv(self.dictionary_path)
            self.map_tgt_to_src = self.df_theme.set_index(self.target_col)[self.source_col].to_dict()
        except Exception as e:
            logger.error(f"Error loading dictionary from {self.dictionary_path}: {e}")
            raise e

    def augment(self):
        logger.info("Applying ReplaceWithSameThemes...")
        expanded_rows = []
        
        for _, row in self.data.iterrows():
            original_tgt = str(row[self.target_col])
            original_src = str(row[self.source_col])
            
            tgt_words = original_tgt.split()
            src_words = original_src.split()
            
            # Logic gốc: Pad source nếu ngắn hơn target (để align index)
            if len(src_words) < len(tgt_words):
                src_words += [""] * (len(tgt_words) - len(src_words))
                
            for i, word_tgt in enumerate(tgt_words):
                if word_tgt in self.map_tgt_to_src:
                    replacement_src = self.map_tgt_to_src[word_tgt]
                    
                    new_tgt_words = tgt_words.copy()
                    new_src_words = src_words.copy()
                    
                    # Thay thế word tại vị trí i
                    # Lưu ý: Logic này giả định Word-to-Word alignment theo index
                    new_tgt_words[i] = word_tgt # Giữ nguyên target? (Code gốc giữ nguyên target, thay source)
                    new_src_words[i] = str(replacement_src)
                    
                    expanded_rows.append([
                        " ".join(new_src_words),
                        " ".join(new_tgt_words)
                    ])
        
        # Kết hợp data gốc + data đã augment
        aug_df = pd.DataFrame(expanded_rows, columns=[self.source_col, self.target_col])
        return pd.concat([self.data, aug_df], ignore_index=True)

class ReplaceWithSameSynonyms(ReplaceWithSameThemes):
    # Logic y hệt Themes, chỉ khác tên gọi và file input (nếu cần)
    def augment(self):
        logger.info("Applying ReplaceWithSameSynonyms...")
        return super().augment()

class RandomInsertion(AugmentMethod):
    def __init__(self, source_col, target_col, data, dictionary_path):
        super().__init__(source_col, target_col, data)
        
        df_theme = pd.read_csv(dictionary_path)
        # Filter 'time' or 'place' themes
        if 'theme' in df_theme.columns:
            filtered = df_theme[df_theme['theme'].isin(['time', 'place'])]
            self.tgt_pool = filtered[self.target_col].dropna().tolist()
            self.src_pool = filtered[self.source_col].dropna().tolist()
        else:
            logger.warning("Column 'theme' not found in dictionary. Using all words.")
            self.tgt_pool = df_theme[self.target_col].dropna().tolist()
            self.src_pool = df_theme[self.source_col].dropna().tolist()

    def augment(self):
        logger.info("Applying RandomInsertion...")
        df = self.data.copy()
        
        def insert_random(text, pool):
            if not pool: return text
            punctuation_pattern = r'([;,!?.])'
            word = random.choice(pool)
            # Chèn từ random trước dấu câu
            return re.sub(punctuation_pattern, f' {word}\\1', str(text))
            
        df[self.target_col] = df[self.target_col].apply(lambda x: insert_random(x, self.tgt_pool))
        df[self.source_col] = df[self.source_col].apply(lambda x: insert_random(x, self.src_pool))
        
        return df

class Deletion(AugmentMethod):
    def __init__(self, source_col, target_col, data, num_deletions=1):
        super().__init__(source_col, target_col, data)
        self.num_deletions = num_deletions

    def augment(self):
        logger.info(f"Applying Deletion (num={self.num_deletions})...")
        deleted_data = []
        
        for _, row in self.data.iterrows():
            words_src = str(row[self.source_col]).split()
            words_tgt = str(row[self.target_col]).split()
            
            # Logic gốc: lặp num_deletions lần
            # Lưu ý: Code gốc của bạn loop len(words) bên trong loop num_deletions -> sinh ra RẤT NHIỀU mẫu
            for _ in range(self.num_deletions):
                for i in range(len(words_src)):
                    if len(words_src) > 1 and len(words_tgt) > 1:
                        new_src = words_src[:]
                        new_tgt = words_tgt[:]
                        
                        idx_src = min(i, len(new_src) - 1)
                        idx_tgt = min(i, len(new_tgt) - 1) # Heuristic align index
                        
                        new_src.pop(idx_src)
                        new_tgt.pop(idx_tgt)
                        
                        deleted_data.append([" ".join(new_src), " ".join(new_tgt)])
                        
        return pd.DataFrame(deleted_data, columns=[self.source_col, self.target_col])

class SlidingWindows(AugmentMethod):
    def __init__(self, source_col, target_col, data, window_size=2):
        super().__init__(source_col, target_col, data)
        self.window_size = window_size

    def augment(self):
        logger.info(f"Applying SlidingWindows (size={self.window_size})...")
        window_data = []
        data_vals = self.data.values
        
        for src, tgt in data_vals:
            w_src = str(src).split()
            w_tgt = str(tgt).split()
            
            if len(w_src) < self.window_size or len(w_tgt) < self.window_size:
                continue
                
            # Logic gốc: Sliding window đồng thời
            for i in range(len(w_src) - self.window_size + 1):
                if i + self.window_size > len(w_tgt):
                    break
                seg_src = " ".join(w_src[i : i + self.window_size])
                seg_tgt = " ".join(w_tgt[i : i + self.window_size])
                window_data.append([seg_src, seg_tgt])
                
        return pd.DataFrame(window_data, columns=[self.source_col, self.target_col])

class DeletionWithOriginal(Deletion):
    def augment(self):
        logger.info("Applying Deletion + Original...")
        deleted_df = super().augment()
        return pd.concat([self.data, deleted_df], ignore_index=True)