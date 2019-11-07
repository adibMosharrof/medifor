import os
import asyncio
import cv2
import numpy as np
import sys
import time

dir = '../../outputs/patches/128_4/target_image'
async def start():
    img_names =os.listdir(dir)
    imgs = await asyncio.gather(*(read_img(img_name) for img_name in img_names))
    return imgs
    
async def read_img(path, error_message=None): 
    path = os.path.join(dir,path)
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No image found at the path {path}")
    except FileNotFoundError as err:
        error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
        raise err
    return image.astype(np.float32)

def read_img1(path, error_message=None): 
    path = os.path.join(dir,path)
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No image found at the path {path}")
    except FileNotFoundError as err:
        error_msg = error_message or 'FAILED to open image: {} \n {} \n {}'.format(path, sys.exc_info()[0], sys.exc_info()[1])
        raise err
    return image.astype(np.float32)
     
def asyncversion():
    t1 = time.perf_counter()
    asyncio.run(start())
    t2 = time.perf_counter()
    print(f'async {round((t2-t1), 3)} seconds')

def vanilla():
    t1 = time.perf_counter()
    img_names =os.listdir(dir)
    imgs = [read_img1(img_name) for img_name in img_names]
    t2 = time.perf_counter()
    print(f'vanilla {round((t2-t1), 3)} seconds')
    return imgs
    
if __name__ == '__main__':
    asyncversion()
    vanilla()
    
    
    