import random
from typing import Dict, List

class DataSplitter:
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, seed: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
    def split_by_shoe_id(self, shoe_ids: List[str]) -> Dict[str, List[str]]:
        random.seed(self.seed)
        shoe_ids = sorted(shoe_ids)
        random.shuffle(shoe_ids)
        
        n_total = len(shoe_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_ids = shoe_ids[:n_train]
        val_ids = shoe_ids[n_train:n_train + n_val]
        test_ids = shoe_ids[n_train + n_val:]
        
        return {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
    
    def get_split_stats(self, split_info: Dict[str, List[str]]) -> Dict:
        stats = {}
        for split_name, shoe_ids in split_info.items():
            stats[split_name] = len(shoe_ids)
        
        total = sum(stats.values())
        stats['total'] = total
        
        for split_name in ['train', 'val', 'test']:
            if split_name in stats:
                stats[f'{split_name}_percentage'] = stats[split_name] / total * 100
        
        return stats