import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

import sys
sys.path.append("D:/Coding/chatBot/src")
from config import Config  # Ensure you have Config.LLAMA_MODEL_NAME

# Load dataset correctly
def load_dataset(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        dataset = Dataset.from_dict({
            "question": [item["question"] for item in data],  
            "answer": [item["answer"] for item in data]
        })
        return dataset.train_test_split(test_size=0.2)  # Split into train and eval sets

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Fine-tune LLaMA
def fine_tune_llama():
    model_name = "meta-llama/Llama-3.2-3B-Instruct" 
    dataset_path = "D:/Coding/chatBot/src/scraped_data.json"
    hf_token = os.getenv("HF_TOKEN")  # Ensure you set this environment variable

    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} not found.")
        return
    
    dataset = load_dataset(dataset_path)
    if dataset is None:
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token if hf_token else True)
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_train_dataset = dataset["train"].map(tokenize_function, batched=True)
    tokenized_eval_dataset = dataset["test"].map(tokenize_function, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token if hf_token else True)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,  # Reduce batch size if you run out of memory
        per_device_eval_batch_size=2,
        save_total_limit=2,
        num_train_epochs=3,
        logging_dir="./logs",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,  # Include eval dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained("./fineTunedLlama")
    tokenizer.save_pretrained("./fineTunedLlama")
    print("Model fine-tuned and saved.")

if __name__ == "__main__":
    fine_tune_llama()
