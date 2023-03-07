import os
import json
train_anno_path = 'LSMDC16_annos_training.csv'
test_anno_path = 'LSMDC16_challenge_1000_publictect.csv'
feature_root = 'lsmdc_3fps_224'

valid_videos = os.listdir(feature_root)
valid_videos_dict = {}
for video in valid_videos:
    if video not in valid_videos_dict:
        valid_videos_dict[video] = 1

blip_anno_train = []
cnt = 0
with open(train_anno_path, 'r') as fp:
    for line in fp:
        line = line.strip()
        line_split = line.split("\t")
        clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
        if clip_id not in valid_videos_dict:
            continue
        item = {'caption':sentence, 'image':clip_id, 'image_id':clip_id}
        blip_anno_train.append(item)
        cnt += 1
json.dump(blip_anno_train, open('train.json','w'))

print(cnt)

blip_anno_test = []
cnt = 0
with open(test_anno_path, 'r') as fp:
    for line in fp:
        line = line.strip()
        line_split = line.split("\t")
        clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
        if clip_id not in valid_videos_dict:
            continue
        item = {'image':clip_id, 'caption':[sentence]}
        blip_anno_test.append(item)
        cnt += 1
json.dump(blip_anno_test, open('test.json','w'))
print(cnt)