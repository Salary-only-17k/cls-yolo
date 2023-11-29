import cv2
import os
import pathlib
import random
import tqdm



def run(pth):
    for f in ['train','val']:
        p_lst = list(pathlib.Path(os.path.join(pth,f)).glob('*.jpg'))
        for p in tqdm.tqdm(p_lst,desc=f):
            p = str(p)
            i1 = random.randint(0,1)
            i2 = random.randint(0,4)
            i3 = random.randint(0,3)
            filename = os.path.basename(p).split(".")[0]+f"{i1}"+f"_{i2}"+f"_{i3}_.jpg"
            os.rename(p,os.path.join(pth,f,filename))
            

if __name__ == "__main__":
    pth = '/media/cheng/data/02-data/cat_dog_dataset/03-dist-mini'
    run(pth)