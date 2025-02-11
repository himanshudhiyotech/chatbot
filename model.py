import json
from datasets import Dataset, load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load and flatten JSON dataset
file_path = "D:/Coding/DhiyoTech/Project/dataset.json"
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

qa_pairs = []
for category in raw_data["About"].values():  # Extracting nested lists
    qa_pairs.extend(category)

# Convert to Hugging Face Dataset
qa_dataset = Dataset.from_list(qa_pairs)
print(f"Loaded dataset with {len(qa_dataset)} examples")

# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Set pad_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or you can use '[PAD]' token if you prefer

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(examples["answer"], truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"],
    }

# Tokenize dataset
tokenized_qa_dataset = qa_dataset.map(tokenize_function, batched=True)
print(f"✅ Tokenized dataset with {len(tokenized_qa_dataset)} examples")

# Split into training and validation sets
train_dataset = tokenized_qa_dataset
val_dataset = train_dataset  # Modify if you want a separate validation set
print(f"✅ Training set size: {len(train_dataset)}")
print(f"✅ Validation set size: {len(val_dataset)}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=500,
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start fine-tuning
if len(train_dataset) > 0:
    trainer.train()
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
    print("✅ Fine-tuning complete! Model saved.")
else:
    print("❌ Training dataset is empty. Please check the data loading process.")






# from datasets import load_dataset

# dataset = load_dataset('json', data_files='D:\Coding\DhiyoTech\Project\dataset.json')

# dataset = dataset['train'].train_test_split(test_size=0.1)
# train_dataset = dataset['train']
# val_dataset = dataset['test']

# from transformers import LlamaForCausalLM, LlamaTokenizer

# model_name = "meta/llama-3.2"  # Replace with the correct model path if different
# model = LlamaForCausalLM.from_pretrained(model_name)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)

# def tokenize_function(examples):
#     return tokenizer(examples['question'], padding="max_length", truncation=True)

# tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_val = val_dataset.map(tokenize_function, batched=True)

# from transformers import TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./results",  # Where to save checkpoints and model
#     evaluation_strategy="steps",  # Evaluation during training
#     num_train_epochs=3,  # Number of epochs
#     per_device_train_batch_size=4,  # Batch size per GPU
#     per_device_eval_batch_size=8,  # Eval batch size
#     logging_dir="./logs",  # Where to save logs
#     logging_steps=500,  # Log every 500 steps
# )

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_val,
# )

# trainer.train()

# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')

# results = trainer.evaluate()
# print(results)

# inputs = tokenizer("What is the meaning of life?", return_tensors="pt")
# outputs = model.generate(inputs["input_ids"], max_length=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))








# from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
# from datasets import Dataset
# import json

# # Load pre-trained Llama tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained("llama-3.2-model")  # Change to actual model name
# model = LlamaForCausalLM.from_pretrained("llama-3.2-model")

# # Load and preprocess the dataset
# def load_dataset():
#     try:
#         with open("D:/Coding/DhiyoTech/Project/dataset.json", "r", encoding="utf-8") as file:
#             return json.load(file)
#     except FileNotFoundError:
#         print("Dataset file not found.")
#         return {}
#     except json.JSONDecodeError:
#         print("Error decoding the dataset JSON.")
#         return {}
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         return {}

# dataset = load_dataset()

# # Function to preprocess dataset into QA pairs
# def preprocess_dataset(dataset):
#     qa_pairs = []
#     for category in dataset.values():
#         for subcategory in category.values():
#             for entry in subcategory:
#                 question = entry["question"]
#                 answer = entry["answer"]
#                 qa_pairs.append({"question": question, "answer": answer})
#     return qa_pairs

# # Preprocess dataset into qa_pairs
# qa_pairs = preprocess_dataset(dataset)

# # Convert dataset into a format suitable for training
# def create_dataset(qa_pairs):
#     # Tokenize the questions and answers
#     encodings = tokenizer([pair['question'] for pair in qa_pairs], truncation=True, padding=True, max_length=512)
#     labels = tokenizer([pair['answer'] for pair in qa_pairs], truncation=True, padding=True, max_length=512)
    
#     # Return dataset in format expected by Trainer
#     return Dataset.from_dict({
#         'input_ids': encodings['input_ids'],
#         'labels': labels['input_ids']
#     })

# # Create the dataset for training
# train_dataset = create_dataset(qa_pairs)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./output",
#     evaluation_strategy="steps",
#     save_steps=500,
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     logging_dir="./logs",
#     logging_steps=100,
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# )

# # Fine-tune the model
# trainer.train()

# # Save the model after fine-tuning
# model.save_pretrained("./llama_model")
# tokenizer.save_pretrained("./llama_model")
