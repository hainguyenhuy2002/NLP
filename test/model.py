import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BARTCNN(nn.Module):
    def __init__(self, max_content_length=1024, max_summary_length=256):
        super(BARTCNN, self).__init__()
        
        # Load the BART encoder
        self.BARTpho = AutoModel.from_pretrained("vinai/bartpho-syllable")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

        for param in self.BARTpho.parameters():
            param.requires_grad = False

        self.num_encoder_hs_use = 4
        self.num_decoder_hs_use = 4
        self.max_content_length = max_content_length
        self.max_summary_length = max_summary_length

        self.encoder_cnn = nn.Sequential(
            # Input: -1 x 4 x 1024 x 1024
            nn.Conv2d(in_channels=self.num_encoder_hs_use, out_channels=2, kernel_size=3, padding=1), 
            nn.ReLU(),
            # -> -1 x 2 x 1024 x 1024
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1), 
            nn.ReLU()
            # -> -1 x 1 x 1024 x 1024
        )

        self.decoder_cnn = nn.Sequential(
            # Input: -1 x 4 x 256 x 1024
            nn.Conv2d(in_channels=self.num_decoder_hs_use, out_channels=2, kernel_size=3, padding=1), 
            nn.ReLU(),
            # -> -1 x 2 x 256 x 1024
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1), 
            nn.ReLU()
            # -> -1 x 1 x 256 x 1024
        )


        self.after_decoder_cnn = nn.Sequential(
            # Input: -1 x 256 x 1024
            nn.Linear(1024, len(self.tokenizer.get_vocab())),
            nn.ReLU(),
            nn.LogSoftmax(dim=2)
            # -> -1 x 256 x vocab_size
        )

        
        
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # Encode the input document
        encoder_outputs = self.BARTpho.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)['hidden_states']
        # encoder_outputs[0].shape = [16, 1024, 1024]
        encoder_outputs = torch.transpose(torch.cat(tuple([hs.unsqueeze(0) for hs in encoder_outputs])[-self.num_encoder_hs_use:], 0), 0, 1)
        encoder_outputs = self.encoder_cnn(encoder_outputs).reshape(-1, self.max_content_length, 1024) # -1 x 1024 x 1024
        # Decode and generate the summary
        
        decoder_outputs = self.BARTpho.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_attention_mask=attention_mask, encoder_hidden_states=encoder_outputs, output_hidden_states=True)['hidden_states']
        decoder_outputs = torch.transpose(torch.cat(tuple([hs.unsqueeze(0) for hs in decoder_outputs])[-self.num_decoder_hs_use:], 0), 0, 1)
        decoder_outputs = self.decoder_cnn(decoder_outputs).reshape(-1, self.max_summary_length, 1024)
       
        return self.after_decoder_cnn(decoder_outputs)
