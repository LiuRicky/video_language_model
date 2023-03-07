import os
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

# running example 

frame_path = ''
video_path = ''

def get_video_ids():
    params = []
    save_path = frame_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for root, dirs, files in os.walk(video_path):
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(save_path, file.split('.')[0])
            params.append((input_path, output_path))
    return params

def extract_frame(params):
    input_path, output_path = params[0], params[1]
    os.makedirs(output_path, exist_ok=True)
    cmd = 'ffmpeg -i {} -s 224*224 -f image2 -vf fps=3 {}/%05d.jpg'.format(input_path, output_path)
    os.system(cmd)

if __name__ == '__main__':
    params = get_video_ids()
    
    # multi process extract video
    workers = min(multiprocessing.cpu_count(), 64)
    pool = Pool(workers)
    res = list(tqdm(pool.imap(extract_frame, params), total=len(params)))


