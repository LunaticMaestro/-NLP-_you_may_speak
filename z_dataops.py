import json
from typing import List
import torch
import unicodedata
from torch.utils.data import DataLoader, Dataset 
from torchvision.transforms import transforms, Lambda
import glob
import os
import string

### Dataset
class NamesDataset(Dataset): 
    '''Loads names from different languages. Store the names in runtime and DOES NOT do lazy loading.
    '''
    def __init__(self, data_dir: str="data/names", transform=None):
        super().__init__()
        # track object variables
        self.data_dir = data_dir 
        self.transform = transform
        # generated variables
        self.names = []
        self.labels = []
        self.classes_to_idx: dict = []
        self.idx_to_classes: dict = []
        
        # locate all languages names file .txt 
        self.read_data_files()
        self.set_classes()

    def read_data_files(self): 
        '''locates files with .txt pattern and reads them, output stored in self.names, labels'''
        files: List[str] = glob.glob(os.path.join(self.data_dir, "*.txt"))
        for file in files: 
            language: str = os.path.splitext(os.path.basename(file))[0]
            # Read File contents
            with open(file, "r") as f:
                contents = f.read() 
                names = contents.split("\n")
            # Store data
            self.names.extend(names)
            self.labels.extend([language for _ in range(len(names))])
        return None

    def __len__(self): 
        return len(self.labels) 
    
    def __getitem__(self, index):
        name = self.names[index] 
        label = self.labels[index] 

        if self.transform: 
            name = self.transform(name) 

        # label: torch.Tensor = torch.zeros((len(self.classes_to_idx)), dtype=torch.float).scatter_(dim=0, index=torch.tensor(self.classes_to_idx.get(label)), value=1)
        label = torch.tensor([self.classes_to_idx.get(label)])
        
        return name.unsqueeze(0), label

    def set_classes(self, cache_location:str = "model/label.json"): 
        '''takes the unique labels and store in self.classes'''
        # first saves the labels to file so it can be used during inferencing 
        unique_labels = list(set(self.labels))

        self.classes_to_idx = dict([(label, i) for i, label in enumerate(unique_labels)])
        self.idx_to_classes = {value: key for key, value in self.classes_to_idx.items()}

        with open(cache_location, "w") as file: 
            json.dump(self.idx_to_classes, file, indent=4)

### Transformations 
## **Why**: So that they can be applied separately during inference

def _allowed_characters(s: str): 
    allowed_characters = string.ascii_letters
    return ''.join([char if allowed_characters.find(char) >= 0 else '' for char in s])

def _unicode_to_ascii(s:str):
    '''Converts Unicode to ASCII to normalize ACCENTS'''
    # CODE from https://stackoverflow.com/a/518232/2809427
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def _string_to_Tensor(name: str): 
    '''Converts to dimensionality (chars, LowerCaseAscii)'''
    name_lower = name
    name_tensor = torch.zeros((len(name_lower), len(string.ascii_letters))).scatter_(dim=1, index= torch.tensor(list(map(string.ascii_letters.index, name_lower))).unsqueeze(1), value=1)
    return name_tensor

transform = transforms.Compose([
    _unicode_to_ascii, 
    _allowed_characters,
    _string_to_Tensor,
])

def proxy_collate_batch(batch: List)-> List[torch.Tensor]: 
    '''Although we are not padding the sequence we created this proxy function to avoid stacking the jagged array.'''
    batch = [(x, y) for x, y in batch if x.shape[1] > 1]
    return batch

if __name__ == "__main__": 
    ds = NamesDataset(transform=transform)
    train_dataset = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=proxy_collate_batch)
    batch = next(iter(train_dataset))
    print(batch[0][0].shape, batch[0][1].shape) # (1, x, 26), # (1)