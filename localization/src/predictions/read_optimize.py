import os
from multiprocessing import Pool
import cv2
import time
import concurrent.futures


dir_path = "../../outputs/patches/128_4/target_image"
dir_path = '../../data/model_sys_predictions/c8-lgb_local_40_nb_a/mask'

size = 100

file_names = os.listdir(dir_path)[:size]

files = [os.path.join(dir_path, fn) for fn in file_names]
# t1 = time.perf_counter()
# for file in files:
#     img = cv2.imread(file)
# t2 = time.perf_counter()
# print(f'Reading Files {round((t2-t1), 3)} seconds')

# t1 = time.perf_counter()
# with Pool(processes=4) as p:
#     print('before reading')
#     print(p.map(cv2.imread, files))
#     print('after reading')
# t2 = time.perf_counter()
# print(f'Reading Files in parallel {round((t2-t1), 3)} seconds')
def myread(path):
    img = cv2.imread(path)
    return hash(tuple(img))

t1 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     res = executor.map(myread, files)
    res = [executor.submit(myread, f) for f in files]



t2 = time.perf_counter()
print(f'Reading Files using threads {round((t2-t1), 3)} seconds')
print(res.shape)

