import numpy as np
from sklearn.metrics import matthews_corrcoef
import time
import concurrent.futures
from multiprocessing import Process, Pool
from functools import partial
import warnings
import sys


class Runner():

  def sleeper(self, pred, man, thr = None):
    return time.sleep(1)

  def mcc_score(self, pred, man, thr = None):
    warnings.filterwarnings("ignore")
    return matthews_corrcoef(pred, man)

  def future_thread(self, func):
    t1 = time.perf_counter()
    for _ in thres:
      with concurrent.futures.ThreadPoolExecutor() as executor:
        results.append(executor.submit(func, pred, man))
    for f in concurrent.futures.as_completed(results):
      res = f.result()

    t2 = time.perf_counter()
    print(f'Future thread {func.__name__} {round((t2-t1), 3)} seconds')
    return

  def future_process(self, func):
    t1 = time.perf_counter()
    for _ in thres:
      with concurrent.futures.ProcessPoolExecutor() as executor:
        results.append(executor.submit(func, pred, man))
    for f in concurrent.futures.as_completed(results):
      res = f.result()

    t2 = time.perf_counter()
    print(f'Future process {func.__name__} {round((t2-t1), 3)} seconds')
    return

  def processes(self, func):
    t1 = time.perf_counter()
    procs = {}
    for t in thres:
      proc = Process(target=func, args=(pred,man))
      procs[t] = proc
      #print(f'Starting processes with threshold {t}')
      proc.start()

    for k, v in procs.items():
      v.join()
      #print(f'Finished Processes with thres {k}' )

    t2 = time.perf_counter()
    print(f'Processes {func.__name__} {round((t2-t1), 3)} seconds')

  def pool(self, func):
    t1 = time.perf_counter()
    p = Pool()
    meth = partial(func, pred, man)
    p.map(meth, thres)
    t2 = time.perf_counter()
    print(f'Pool {func.__name__} {round((t2-t1), 3)} seconds')

  def vanilla(self, func):
    t1 = time.perf_counter()
    for t in thres:
      #print(f'Starting vanilla with threshold {t}')
      func(pred, man)
      #print(f'Finished vanilla with thres {t}' )
    t2 = time.perf_counter()
    print(f'vanilla {func.__name__} {round((t2-t1), 3)} seconds')


if __name__== "__main__":
    #warnings.filterwarnings("ignore")
    print(sys.version)
    r = Runner()
    thres = np.arange(0,1, 0.3)
    print(f"Number of thresholds {len(thres)}")
    pred = [1]*200000
    man = [1]*200000
    results = []
    
    r.processes(r.mcc_score)
    r.pool(r.mcc_score)
    r.future_thread(r.mcc_score)
    r.future_process(r.mcc_score)
    r.vanilla(r.mcc_score)
    
#     r.processes(r.sleeper)
#     r.pool(r.sleeper)
#     r.future_thread(r.sleeper)
#     r.future_process(r.sleeper)
#     r.vanilla(r.sleeper)
    
    

