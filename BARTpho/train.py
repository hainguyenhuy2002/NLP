import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, MBartForConditionalGeneration
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch, argparse, os

parser = argparse.ArgumentParser(description='Model Live')
parser.add_argument("--train_data", default="", type=str)
parser.add_argument("--checkpoint_path", default="", type=str)
args = parser.parse_args()

assert len(args.checkpoint_path) > 0 and len(args.train_data), "You have to pass checkpoint_path and train_data" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable") 
model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable").to(device)
df_train = pd.read_csv(args.train_data)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["labels"], max_length=256, truncation=True, padding=True
        )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs

df_train = df_train[["original", "summary"]]
df_train.rename(columns = {'original':'inputs'}, inplace = True)
df_train.rename(columns = {'summary':'labels'}, inplace = True)

dataset = Dataset.from_pandas(df_train)
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

training_args = Seq2SeqTrainingArguments(
    args.checkpoint_path, do_train=True, do_eval=False, num_train_epochs=5,
    learning_rate=1e-5, warmup_ratio=0.05, weight_decay=0.01, per_device_train_batch_size=1,
    per_device_eval_batch_size=1, logging_dir='./log', group_by_length=True,
    save_strategy="epoch", save_total_limit=3, fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

print("Start training")
if (len(os.listdir(args.checkpoint_path) > 0)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()