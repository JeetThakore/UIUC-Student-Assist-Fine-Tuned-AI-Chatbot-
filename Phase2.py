
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

DATA_PATH = r"C:\Users\91940\Downloads\Database_Dump\Medibot - Extension\Data"

print("Loading PDFs...")
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"✅ Loaded {len(documents)} PDFs")

print("Splitting PDFs into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(text_chunks)} text chunks")

# Take a slightly larger subset
text_chunks = text_chunks[:30]  # 30 chunks
print(f"Using {len(text_chunks)} chunks for fine-tuning (~20 mins)")

print("Preparing training data...")
def prepare_training_data(text_chunks):
    training_data = []
    for chunk in tqdm(text_chunks, desc="Generating QA pairs"):
        context = chunk.page_content.strip()
        if len(context) > 100:
            question = "Summarize the following university information:"
            answer = context[:300]  # moderate answer length
            training_data.append({"context": context, "question": question, "answer": answer})
    return training_data

training_data = prepare_training_data(text_chunks)
print(f"✅ Prepared {len(training_data)} training examples")

dataset = Dataset.from_dict({
    "context": [d["context"] for d in training_data],
    "question": [d["question"] for d in training_data],
    "answer": [d["answer"] for d in training_data]
})

print("Loading model and tokenizer...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded")

print("🔁 Tokenizing dataset...")
def preprocess_function(examples):
    inputs = [f"Context: {c}\nQuestion: {q}" for c, q in zip(examples["context"], examples["question"])]
    targets = [a for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=320, truncation=True)  # slightly longer input
    labels = tokenizer(targets, max_length=320, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "answer"])
print("✅ Dataset tokenized")

print("📦 Setting up trainer and training arguments...")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./fine_tuned_flan_uiuc_20min",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=5,
    fp16=False,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting fine-tuning (~20 mins expected)...")
trainer.train()
print("✅ Fine-tuning complete")

print(" Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_flan_uiuc_20min")
tokenizer.save_pretrained("./fine_tuned_flan_uiuc_20min")
print(" Model saved at './fine_tuned_flan_uiuc_20min'")


