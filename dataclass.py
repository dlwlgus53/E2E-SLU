# read text and audio

# text -> vectorize
# audio -> vectorize

# vectorize model is fixed size

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model, logging
logging.set_verbosity_error()


class Wav2Vec2(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.w2v2_ckpt_path = hp["hgf_w2v2_ckpt"] # hgf_w2v2_ckpt: "wav2vec2/k-wav2vec_huggingface"

    def forward(self, inputs):
        model = Wav2Vec2Model.from_pretrained(self.w2v2_ckpt_path, local_files_only=True).cuda()
        for param in model.parameters():
            param.requires_grad = False
            param.grad = None
        model.eval()
        # print("model:", model)

        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
        y = outputs.hidden_states[12]
        # y = torch.tensor(y, dtype=torch.float32)
        # y = y.permute((0,2,1)) # 16, w2v2_final_layer_norm: 768, diff. num
        # y <- 16, diff num, 768
        return y