import os
import shutil
import random

source_yes = r'C:\Users\mukes\Downloads\archive (1)\yes'
source_no  = r'C:\Users\mukes\Downloads\archive (1)\no'

train_ratio = 0.7
valid_ratio = 0.15
test_ratio  = 0.15

def split_and_copy(source_folder, class_name):
    images = [f for f in os.listdir(source_folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    splits = {
        'train': images[:train_end],
        'valid': images[train_end:valid_end],
        'test' : images[valid_end:]
    }
    
    for split_name, files in splits.items():
        dest = os.path.join('dataset', split_name, class_name)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(source_folder, f), dest)
        print(f"{split_name}/{class_name}: {len(files)} images")

print("Splitting dataset...")
split_and_copy(source_yes, 'yes')
split_and_copy(source_no, 'no')
print("\nDataset ready! ✅")