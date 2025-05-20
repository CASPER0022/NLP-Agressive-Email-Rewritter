from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
import torch

# Load full dataset
dataset = load_dataset("jdustinwind/Polite")["train"]

# Select 1000 samples for training and 200 for evaluation
small_train_dataset = dataset.select(range(1000))
small_eval_dataset = dataset.select(range(1000, 1200))

# Initialize tokenizer and add special token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default
tokenizer.add_special_tokens({'sep_token': '<|sep|>'})

# Format examples
def format_example(example):
    return {
        "text": f"Rewrite this politely: {example['src']} <|sep|> {example['tgt']}",
        "labels": f"Polite version: {example['tgt']}"
    }

small_train_dataset = small_train_dataset.map(format_example)
small_eval_dataset = small_eval_dataset.map(format_example)

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        text_target=batch["labels"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

small_train_dataset = small_train_dataset.map(tokenize, batched=True)
small_eval_dataset = small_eval_dataset.map(tokenize, batched=True)

# Set PyTorch format
small_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
small_eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model and resize embeddings for special token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-polite",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("fine-tuned-gpt2-polite")
tokenizer.save_pretrained("fine-tuned-gpt2-polite")

# Generation pipeline
generator = pipeline(
    "text-generation",
    model="fine-tuned-gpt2-polite",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Generate example polite rewriting
prompt = "Rewrite this politely: Why are you always late to work? <|sep|>"
output = generator(
    prompt,
    max_length=128,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

print("Generated:", output[0]['generated_text'])
