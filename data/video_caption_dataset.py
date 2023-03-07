import os
import json
import copy

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class video_caption_train(Dataset):
    def __init__(self, transform, video_root, ann_root, max_words=30, prompt='', max_img_size=224, num_frm=1):        
        '''
        video_root (string): Root directory of video frm (e.g. msrvtt/frms/)
        ann_root (string): directory to store the annotation file
        '''
        filename = 'train.json'
        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        # ann_len = int(len(self.annotation)*0.20)
        # self.annotation = self.annotation[:ann_len]
        self.transform = transform
        self.video_root = video_root
        self.max_words = max_words      
        self.prompt = prompt
        self.max_img_size = max_img_size
        self.num_frm = num_frm
        
        self.video_ids = {}  
        n = 1
        for ann in self.annotation:
            video_id = ann['video']
            if video_id not in self.video_ids.keys():
                self.video_ids[video_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        video_path = os.path.join(self.video_root, ann['video'])
        if 'ts' in ann:
            video = load_video_from_path_decord(video_path, self.transform, height=self.max_img_size, width=self.max_img_size, start_time=ann['ts'][0], end_time=ann['ts'][1], fps=3, num_frm=self.num_frm)
        else:
            video = load_video_from_path_decord(video_path, self.transform, height=self.max_img_size, width=self.max_img_size, num_frm=self.num_frm)      
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return video, caption, self.video_ids[ann['video']] 
    
    
class video_caption_eval(Dataset):
    def __init__(self, transform, video_root, ann_root, split, max_img_size=224, num_frm=1):  
        '''
        video_root (string): Root directory of video frm (e.g. msrvtt/frms/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'test.json','test':'test.json'}
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.video_root = video_root
        self.max_img_size = max_img_size
        self.num_frm = num_frm
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        video_path = os.path.join(self.video_root, ann['video']) 
        if 'ts' in ann:
            video = load_video_from_path_decord(video_path, self.transform, height=self.max_img_size, width=self.max_img_size, start_time=ann['ts'][0], end_time=ann['ts'][1], fps=3, 
                                                num_frm=self.num_frm, frm_sampling_strategy='uniform')    
        else:
            video = load_video_from_path_decord(video_path, self.transform, height=self.max_img_size, width=self.max_img_size, 
                                                num_frm=self.num_frm, frm_sampling_strategy='uniform')       
        return video, ann['video_id']

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

if __name__ == '__main__':
    video_path = '/cfs/cfs-rmuhzak3/lukenxu/dataset/msvd/frames_fps5_224/eyhzdC936uk_15_27'
    video = load_video_from_path_decord(video_path, None, height=224, width=224, 
                                            num_frm=1, frm_sampling_strategy='uniform') 
