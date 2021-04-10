import config
import glob
import os
import pandas as pd

from sklearn.model_selection import train_test_split

def create_csv(path_dataset,train=True):
    folders = os.listdir(path_dataset)
    path = []
    labels = []
    for folder in folders:
        # print(folders)
        folder_path = os.path.join(path_dataset,folder)
        files = glob.glob(folder_path+"/*.jpg")
        for f in files:
            # print(f,folder)
            path.append(f)
            labels.append(config.MAPPING[folder])
    data = {
        'path':path,
        'label':labels
    }

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = create_csv(config.TRAIN_IMAGES)
    # print(len(df))
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=0, stratify=df['label'], shuffle=True)
    df_train.to_csv(config.TRAIN_CSV)
    df_val.to_csv(config.VAL_CSV)
    print(len(df_train),len(df_val))
    print(df_train.nunique(),df_val.nunique())
