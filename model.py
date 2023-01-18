# not pretraind transformer
# should make audio and text encoder size same
# input : concat the vec(text) and vec(audio)
# output : DST value

# the below is sample code of T5 modeling
#  (by the way,,, As I know, t5 model structure is same with tranformer(enc-dec), so I think.. it doesn't matter to use T5==transformer.)

import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
import os


"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
import pdb
# from transformers import AutoModel, AutoTokenizer

class Text_Encoder(nn.Module):
    def __init__(self, bert_model, tokenizer, class_num):
        super(Text_Encoder, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        
        # Instance-CL head
        self.classification_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, class_num))
        
    
    def forward(self, input_ids, attention_mask):
        embeded = self.get_mean_embeddings( input_ids,attention_mask)
        output= self.classification_head(embeded)
        return output

    
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output
    



class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, special_token_list, dropout, add_special_decoder_token, is_training):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.special_token_list = special_token_list
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
        self.sos_r_token_id, self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
            '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>'])
        if is_training:
            print ('Initializing Huggingface T5 model...')
            t5_config = T5Config.from_pretrained(model_path)
            t5_config.__dict__["dropout"] = dropout
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        else:
            print ('Loading Model from pretrained ckpt...')
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.add_special_decoder_token = add_special_decoder_token

    ## THIS IS IMPORTANT PART ##
    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type()) 
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss


    def batch_generate(self, src_input, src_mask, generate_mode, max_decode_len, return_confidence = False):
        '''
            This function deals with batch generation. In order to fully take advantage of batch inference,
            in each batch, we only generate one type of output. e.g. Given a batch of dialogue history, we 
            generate the corresponding belief state/dialogue action/system response for the given batch. The 
            specific type of output is decided by the input argument "generate_mode"
        '''
        if self.add_special_decoder_token:
            if generate_mode == 'bs':
                start_token, end_token, start_token_id, end_token_id = '<sos_b>', '<eos_b>', self.sos_b_token_id, self.eos_b_token_id
            elif generate_mode == 'da':
                start_token, end_token, start_token_id, end_token_id = '<sos_a>', '<eos_a>', self.sos_a_token_id, self.eos_a_token_id
            elif generate_mode == 'nlg':
                start_token, end_token, start_token_id, end_token_id = '<sos_r>', '<eos_r>', self.sos_r_token_id, self.eos_r_token_id
            else:
                raise Exception('Wrong Generate Mode!!!')
        else:
            start_token, end_token = '<pad>', '</s>'
            start_token_id, end_token_id = \
            self.tokenizer.convert_tokens_to_ids([start_token])[0], self.tokenizer.convert_tokens_to_ids([end_token])[0]


        origin_outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=src_input)
        max_logit = torch.max(origin_outputs.logits,2).values

        outputs = self.model.generate(input_ids = src_input, attention_mask = src_mask, decoder_start_token_id = start_token_id,
            pad_token_id = self.pad_token_id, eos_token_id = end_token_id, max_length = max_decode_len)

        res_text_list = []
        confidence_list = []
        for predicted_ids, logit in zip(outputs, max_logit):
            one_res_text = self.tokenized_decode(predicted_ids)
            #print (one_res_text)
            one_res_text = one_res_text.split(start_token)[-1].split(end_token)[0].strip()

            try:
                first_idx = (predicted_ids.cpu() == 32112).nonzero(as_tuple=True)[0][0] 
            except:
                first_idx = 0
            try:
                last_idx = (predicted_ids.cpu() == 32100).nonzero(as_tuple=True)[0][0] + 1
            except:
                last_idx = len(predicted_ids)

            confidence = torch.mean(logit[first_idx:last_idx]).item()
            confidence_list.append(confidence)

            final_res_list = []
            for token in one_res_text.split():
                if token == '<_PAD_>':
                    continue
                else:
                    final_res_list.append(token)
            one_res_text = ' '.join(final_res_list).strip()
            
            res_text_list.append(one_res_text)

        if return_confidence:
            return res_text_list, confidence_list
        else:
            return rest_text_list

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
        


