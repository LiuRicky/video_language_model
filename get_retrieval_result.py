import numpy as np
import json

def itm_eval(scores_t2i, anno):
    cnt = 0
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    right_result_dict = {}
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # print(inds)
        if inds[0] == index:
            cnt += 1
            # print(index)
            # print(anno[index])
            right_result_dict[index] = anno[index]
    print(cnt)
    return right_result_dict

def get_diff_index(base_result_dict, new_result_dict):
    diff_items = {}
    cnt = 0
    for k, v in new_result_dict.items():
        if k not in base_result_dict:
            diff_items[k] = v
            cnt += 1

    return diff_items, cnt

if __name__ == '__main__':
    scores = np.load('/cfs/cfs-rmuhzak3/jarvicliu/blip_adapter/output/retrieval_msrvtt_test/retrieval_result_origin.npy')
    anno = json.load(open('annotation/msrvtt_retrieval/test.json'))
    base_result_dict = itm_eval(scores, anno)
    # print(base_result_dict)

    scores = np.load('/cfs/cfs-rmuhzak3/jarvicliu/blip_adapter/output/retrieval_msrvtt_test/retrieval_result_token_mix.npy')
    new_result_dict = itm_eval(scores, anno)
    print(new_result_dict)

    # diff_items, cnt = get_diff_index(new_result_dict, base_result_dict)
    # print(diff_items)
    # print(cnt)

    diff_items, cnt = get_diff_index(base_result_dict, new_result_dict)
    print(diff_items)
    print(cnt)
    json.dump(diff_items, open('/cfs/cfs-rmuhzak3/jarvicliu/blip_adapter/output/retrieval_msrvtt_test/diff_result.json', 'w'))