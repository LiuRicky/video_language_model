import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

class videoqa_dataset(Dataset):
    def __init__(self, transform, ann_root, video_root, train_files=[], split="train", max_img_size=224, num_frm=1):
        self.split = split        

        self.transform = transform
        self.video_root = video_root
        self.max_img_size = max_img_size
        self.num_frm = num_frm
        
        if split=='train':
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
            self.annotation = json.load(open(os.path.join(ann_root,'test.json'),'r'))    
            
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        video_path = os.path.join(self.video_root,ann['video_id'])
            
        image = load_video_from_path_decord(video_path, self.transform, height=self.max_img_size,
                                            width=self.max_img_size, num_frm=self.num_frm)        
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'])        
            
            # if ann['dataset']=='vqa':               
            #     answer_weight = {}
            #     for answer in ann['answer']:
            #         if answer in answer_weight.keys():
            #             answer_weight[answer] += 1/len(ann['answer'])
            #         else:
            #             answer_weight[answer] = 1/len(ann['answer'])

            #     answers = list(answer_weight.keys())
            #     weights = list(answer_weight.values())

            # elif ann['dataset']=='vg':
            answers = [ann['answer']]
            weights = [0.2]  

            return image, question, answers, weights
        
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        

def load_video_from_path_decord(video_path, transform, height=None, width=None, start_time=None, end_time=None, fps=-1,
                                num_frm=1, frm_sampling_strategy='rand'):
    
    def expand_video_frms(video_frms):
        video_frms_clone = copy.deepcopy(video_frms)
        video_frms_ret = []
        for i in range(len(video_frms)):
            video_frms_ret.append(video_frms[i])
            video_frms_ret.append(video_frms_clone[i])
        return video_frms_ret

    try:
        video_frms = os.listdir(video_path)
        video_frms.sort()
        vlen = len(video_frms)

        if start_time or end_time:
            assert fps > 0, 'must provide video fps if specifying start and end time.'

            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)
            if start_idx < end_idx:
                video_frms = video_frms[start_idx:end_idx]

        # append frames when less
        while(len(video_frms) <= num_frm):
            video_frms = expand_video_frms(video_frms)
        vlen = len(video_frms)
        
        start_idx, end_idx = 0, vlen

        if frm_sampling_strategy == 'uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm, dtype=int)
            # frame_indices = np.linspace(start_idx, end_idx-1, num_frm, dtype=int)
        elif frm_sampling_strategy == 'rand':
            frame_indices = sorted(random.sample(range(vlen), num_frm))
        elif frm_sampling_strategy == 'headtail':
            frame_indices_head = sorted(random.sample(range(vlen // 2), num_frm // 2))
            frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), num_frm // 2))
            frame_indices = frame_indices_head + frame_indices_tail
        else:
            raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

        # raw_sample_frms = vr.get_batch(frame_indices) # (num_frm, height, weight, channel)

        # pre-process frames
        images = []
        for index in frame_indices:
            image_path = os.path.join(video_path, video_frms[index])
            images.append(transform(Image.open(image_path).convert("RGB"))) # (num_frm, channel, height, weight)
            # images.append(Image.open(image_path).convert("RGB"))

        # convert into tensor
        if len(images) > 0:
            raw_sample_frms = torch.tensor(np.stack(images))
        else:
            raw_sample_frms = torch.zeros(1)

    except Exception as e:
        print(e)
        return None

    # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # (num_frm, channel, height, weight)

    return raw_sample_frms