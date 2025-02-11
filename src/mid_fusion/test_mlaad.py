from utils.datasets import MLAADEnDataset, custom_collate
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
ds = MLAADEnDataset(root_dir='/data/amathur-23/DADA/MLAADv5', split='val', sampling_rate=16000, max_duration=4)
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=custom_collate)
from tqdm import tqdm
for batch in tqdm(loader): continue
