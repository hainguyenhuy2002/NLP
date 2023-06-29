from transformers import MBartForConditionalGeneration, BartphoTokenizer, pipeline
import torch, evaluate, string, json, tqdm, os, gc, argparse
from datasets import Dataset
import torch.nn as nn
import pandas as pd

parser = argparse.ArgumentParser(description='Model Live')
parser.add_argument("--train_data", default="", type=str)
parser.add_argument("--dev_data", default="", type=str)
parser.add_argument("--test_data", default="", type=str)
parser.add_argument("--checkpoint_path", default="", type=str)
parser.add_argument("--save_folder", default="", type=str)
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
rouge = evaluate.load('rouge')

tokenizer = BartphoTokenizer.from_pretrained("vinai/bartpho-syllable") 
model = MBartForConditionalGeneration.from_pretrained(args.checkpoint_path).to(device)

save_folder = args.save_folder
num_generate = 16
all_df = pd.concat([
    pd.read_csv(args.train_data),
    pd.read_csv(args.test_data),
    pd.read_csv(args.dev_data)
])


all_df.drop("Unnamed: 0", axis=1, inplace=True)
all_df['index'] = [i for i in range(len(all_df))]
all_df.reset_index(drop=True, inplace=True)


def post_processing(text):
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")
    text = text.replace("<unk>", " ")
    text = text.replace("<pad>", " ")
    text = text.replace("_", " ")
    text = " ".join(text.strip().split()).lower()
    return text


def summary_text_func(generator, row):
    index = row['index']
    output_path = os.path.join(save_folder, f"{index}.json")
    if os.path.exists(output_path):
        return
    original_text = post_processing(row['original'])
    summary_text = post_processing(row['summary'])
    
    # Generate text using the pipeline
    outputs = generator(original_text)
    
    dict_save = dict()
    dict_save["article_untok"] = original_text
    dict_save["abstract_untok"] = summary_text
    dict_save["candidates_untok"] = []
    
    for output in outputs:
        output = post_processing(tokenizer.decode(
            output['generated_token_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ))
        rouge_1_score = rouge.compute(predictions=[output], references=[summary_text])['rouge1']
        dict_save["candidates_untok"].append([output, rouge_1_score])

    torch.cuda.empty_cache()
    gc.collect()
    
    json_object = json.dumps(dict_save, indent = 4)
    with open(output_path, "w") as f:
        f.write(json_object)


# dataset = Dataset.from_pandas(all_df)
with torch.no_grad():
    generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    framework="pt",  
    max_length=1024,  
    num_beams=20,  # Number of beams for beam search
    num_return_sequences=16,  # Number of final sequences to return
    diversity_penalty=0.8,  # Diversity penalty
    do_sample=False,  # Enable sampling
    num_beam_groups = 20,
    temperature=1,
    truncation=True, return_tensors=True, return_text=False)
    all_df.apply(lambda example: summary_text_func(generator, example), axis=1)