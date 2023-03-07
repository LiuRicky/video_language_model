import pandas as pd 
import json
import os

def gen_blip_anno(df):
    
    df["id"] = df.id.apply(int)
    df["video_id"] = df.video_id.apply(int)
    df["video_id"] = df.video_id.apply(lambda x: "video{}".format(x))

    df.rename(columns={'id': 'question_id'}, inplace=True)
    df.drop(['category_id', 'type'], axis=1, inplace=True)
    df_json = df.to_json(orient='records', force_ascii=False)
    return df_json


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    train_df.dropna(inplace=True)

    # print(train_df.info())
    val_df = pd.read_csv('val.csv')
    # print(val_df.info())
    train_df = pd.concat([train_df, val_df], axis=0)
    # print(train_df.info())
    train_json = gen_blip_anno(train_df)
    # print(train_json)
    train_json = json.loads(train_json)
    with open('../train.json', 'w') as f:
        json.dump(train_json, f)

    test_df = pd.read_csv('test.csv')
    test_json = gen_blip_anno(test_df)
    test_json = json.loads(test_json)
    with open('../test.json', 'w') as f:
        json.dump(test_json, f)