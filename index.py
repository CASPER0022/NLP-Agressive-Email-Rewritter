from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# Set all random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# 1. Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9,.!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. Data Loading and Processing
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return Dataset.from_pandas(df)

# 3. Load and Tokenize Data
dataset = load_data('email_tone.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(
        batch['cleaned_text'], 
        padding='max_length', 
        truncation=True,
        max_length=128
    )

tokenized_data = dataset.map(tokenize, batched=True)
tokenized_data = tokenized_data.train_test_split(test_size=0.2)

# 4. Model Initialization with deterministic settings
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)
model.eval()  # Set model to evaluation mode

# 5. Training Configuration
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    warmup_steps=500,
    save_total_limit=1,
    seed=SEED,  # Add seed to training arguments
)

# 6. Metrics Calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

# 7. Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics
)

# 8. Training
print("Starting training...")
trainer.train()
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
print("Training completed!")

# 9. Deterministic Prediction Function
def predict_tone(text):
    # Ensure consistent tokenization
    inputs = tokenizer(
        clean_text(text),
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )
    
    # Disable all random operations
    with torch.no_grad():
        model.eval()
        outputs = model(**inputs)
    
    # Get the most confident prediction
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    
    tones = ['Neutral', 'Passive-Aggressive', 'Aggressive', 'Polite']
    return tones[pred_idx]

# Test consistency
 # Should always return 'Aggressive'
 # Should be identical