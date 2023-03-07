'''
File: blip_video_caption.py
Author: Yuqi Liu
'''
from models.med import BertConfig, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
from models.video_swin_transformer import create_video_swin_transformer
from models.vit import Block

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import time

class BLIP_Video_Caption(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.temporal_transformer = TemporalTransformer(vision_width)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, video, caption):
        # for vit
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H) # shape is (B*N, C, W, H)
        video_embeds = self.visual_encoder(video) # shape is (B*N, patch_len, hidden_dim)
        video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]).view(B,-1,video_embeds.shape[-1])  # shape is (B, N*patch_len, hidden_dim)

        # # ### temp attn ###
        # video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]) # shape is (B, N, patch_len, hidden_dim)
        # video_embeds = video_embeds.permute(0,2,1,3).reshape(-1, N, video_embeds.shape[-1]) # shape is (B*patch_len, N, hidden_dim)
        # video_embeds = self.temporal_transformer(video_embeds) # global temporal trans
        # video_embeds = video_embeds.view(B, -1, N, video_embeds.shape[-1]).view(B, -1, video_embeds.shape[-1]) # shape is (B, N*patch_len, hidden_dim)
        # ### temp attn ###

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(video.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = video_embeds,
                                           encoder_attention_mask = video_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
        
    def generate(self, video, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # for vit
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H) # shape is (B*N, C, W, H)
        video_embeds = self.visual_encoder(video) # shape is (B*N, patch_len, hidden_dim)
        video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]).view(B,-1,video_embeds.shape[-1])  # shape is (B, N*patch_len, hidden_dim)

        # ### temp attn ###
        # video_embeds = video_embeds.view(B,N,video_embeds.shape[-2],video_embeds.shape[-1]) # shape is (B, N, patch_len, hidden_dim)
        # video_embeds = video_embeds.permute(0,2,1,3).reshape(-1, N, video_embeds.shape[-1]) # shape is (B*patch_len, N, hidden_dim)
        # video_embeds = self.temporal_transformer(video_embeds) # global temporal trans
        # video_embeds = video_embeds.view(B, -1, N, video_embeds.shape[-1]).view(B, -1, video_embeds.shape[-1]) # shape is (B, N*patch_len, hidden_dim)
        # ### temp attn ###

        if not sample:
            video_embeds = video_embeds.repeat_interleave(num_beams,dim=0)
            
        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(video.device)
        model_kwargs = {"encoder_hidden_states": video_embeds, "encoder_attention_mask":video_atts}
        
        prompt = [self.prompt] * B
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(video.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions


class TemporalTransformer(nn.Module):
    def __init__(self, vision_width=384):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=vision_width, num_heads=12
            )
            for i in range(4)])
    def forward(self, x):
        for i,blk in enumerate(self.blocks):
            x = blk(x)
        return x

def blip_video_caption(pretrained='',**kwargs):
    model = BLIP_Video_Caption(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print(msg.missing_keys)
        # assert(len(msg.missing_keys)==0)
    return model