import os
import cv2
import time
from multiprocessing import Pool, freeze_support


def read_all(dir_path, starting_index, ending_index):
    img_names = os.listdir(dir_path)[starting_index:ending_index]
    imgs = []
    for name in img_names:
        img_path = os.path.join(dir_path, name)
        img = read(img_path)
        imgs.append(img)
    return imgs

def read_all_mult(dir_path, starting_index, ending_index):
    img_names = os.listdir(dir_path)[starting_index:ending_index]
    with Pool() as p:
        imgs = p.map(read, img_names)
    print(len(imgs))

def read(path):
    return cv2.imread(path)
    

if __name__ == '__main__':
    freeze_support()
    starting_index = 0
    ending_index = 60
    dir_path = 'C:/MyFiles/Study/research/localization/outputs/patches/128_1/target_image'
    
    t1 = time.perf_counter()
    read_all(dir_path,starting_index, ending_index)
    t2 = time.perf_counter()
    print(f'vanilla {round((t2-t1), 3)} seconds')
    
    t1 = time.perf_counter()
    read_all_mult(dir_path,starting_index, ending_index)
    t2 = time.perf_counter()
    print(f'mult {round((t2-t1), 3)} seconds')