import torch
import torch.nn as nn


class HFModel(nn.Module):
    """wrapper for huggingface models to work with our training loop"""

    def __init__(self, model, embedder, head, cfg):
        super().__init__()
        self.model = model
        self.embedder = embedder
        self.head = head
        self.is_encoder_decoder = self.check_is_encoder_decoder()
        self.cfg = cfg
        print("This model is an encoder-decoder model: ", self.is_encoder_decoder)
        print(f"\nInitializing Huggingface model with config:\n{model.config}\n")

    def forward(self, inputs_embeds):
        if self.is_encoder_decoder:
            if "pass_embed" in self.cfg.model and self.cfg.model.pass_embed == True:
                output = self.model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=inputs_embeds
                )
            else:
                dummy_input = torch.ones(
                    [inputs_embeds.size(0), 1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                decoder_input_ids = self.model._shift_right(dummy_input)
                output = self.model(
                    inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids
                )
        else:
            output = self.model(inputs_embeds=inputs_embeds)
        return output.last_hidden_state  # [batch_size, seq_len, hidden_size]

    def check_is_encoder_decoder(self):
        encoder_decoder_list = ["t5"]
        return self.model.config.model_type in encoder_decoder_list
