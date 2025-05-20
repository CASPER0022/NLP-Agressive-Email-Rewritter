from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

# Load model and tokenizer
model_dir = "fine-tuned-gpt2-polite"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate polite output
def generate_polite(text):
    input_text = f"Rewrite this politely: {text} <|sep|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.replace(input_text, "").strip()

# Try some examples
examples = [
    "Move aside.",
    "You are wrong.",
    "That is a terrible idea."
]

for ex in examples:
    polite = generate_polite(ex)
    print(f"\nOriginal: {ex}\nPolite: {polite}")
