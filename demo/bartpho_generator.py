from transformers import MBartForConditionalGeneration, BartphoTokenizer, pipeline
import string

class BartphoGenerator:

    def __init__(self, pretrain_path, device, temperature=1):
        self.tokenizer = BartphoTokenizer.from_pretrained("vinai/bartpho-syllable")
        self.model = MBartForConditionalGeneration.from_pretrained(pretrain_path).to(device)
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            framework="pt",
            max_length=1024,
            num_beams=20,  # Number of beams for beam search
            num_return_sequences=16,  # Number of final sequences to return
            diversity_penalty=0.8,  # Diversity penalty
            do_sample=False,  # Enable sampling
            num_beam_groups = 20,
            temperature=temperature,
            truncation=True, return_tensors=True, return_text=False
        )

    def post_processing(self, text):
        text = text.replace("</s>", " ")
        text = text.replace("<s>", " ")
        text = text.replace("<unk>", " ")
        text = text.replace("<pad>", " ")
        text = text.replace("_", " ")
        for punc in string.punctuation:
            text = text.replace(punc, f" {punc} ")
        split_text = text.strip().split()
        split_text = [t for t in split_text if len(t) > 0]
        text = " ".join(split_text).lower()
        return text
    
    def generate_candidates(self, origin_text):
        original_text = self.post_processing(origin_text)
        outputs = self.generator(original_text)
        candidates = []
        for output in outputs:
            output = self.post_processing(self.tokenizer.decode(
                output['generated_token_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ))
            candidates.append(output)
        return candidates
