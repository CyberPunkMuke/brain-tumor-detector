import os
import sys
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Path fix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features

def load_dataset(folder_path):
    X, y = [], []
    for label, class_name in enumerate(['no', 'yes']):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.exists(class_path):
            print(f"Folder not found: {class_path}")
            continue
        files = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Loading {len(files)} images from {class_path}...")
        for img_file in files:
            img_path = os.path.join(class_path, img_file)
            try:
                features = extract_features(img_path)
                X.append(features[0])
                y.append(label)
            except Exception as e:
                print(f"Skipping {img_file}: {e}")
    return np.array(X), np.array(y)

# Base dataset path
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')

print("Loading train data...")
X_train, y_train = load_dataset(os.path.join(base_path, 'train'))

print("\nLoading validation data...")
X_valid, y_valid = load_dataset(os.path.join(base_path, 'valid'))

print("\nLoading test data...")
X_test, y_test = load_dataset(os.path.join(base_path, 'test'))

print(f"\nTrain: {len(X_train)} | Valid: {len(X_valid)} | Test: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)

# Test accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['No Tumor', 'Tumor']))

# Save model
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tumor_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved! ✅")