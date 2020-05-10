from math import sqrt
import sys, os
sys.path.append('..')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import numpy as np
import time
from shared.image_utils import ImageUtils

    
class MccBinarized():    
    # Dummy data to process and compute MCC
 
    # Compute Matthew's correlation coefficient based on definition (see, e.g., Wikipedia)
    @staticmethod
    def mcc(tt, tf, ft, ff):
        numer = (tt*ff - tf*ft)
        denom = sqrt((tt+tf)*(tt+ft)*(ff+tf)*(ff+ft))
        if denom > 0:
            return numer/denom
        else:
            return 0.0

    @staticmethod
    def compute(prediction, ground_truth):
        # NOTE: These must be set to the number of possible thresholds, assuming the first threshold is 0.
        max_score = 255
        num_scores = max_score + 1
    
        # Keep track of the number of true and false examples at each prediction value
        counts_t = [0] * num_scores
        counts_f = [0] * num_scores
    
        # Process each prediction in turn
        for i in range(len(ground_truth)):
            pred = int(prediction[i])
    
            # Assuming truth is an int. 0 = false; 1 = true
            if ground_truth[i] > 0:
                counts_t[pred] = counts_t[pred] + 1
            else:
                counts_f[pred] = counts_f[pred] + 1
    
        # Helpful summary variables
        num_true = sum(counts_t)
        num_false = sum(counts_f)
        n = num_true + num_false
    
        # Initialize values for MCC stat computations
        mcc_val = [0] * num_scores
        total_tt = num_true
        total_tf = num_false
        total_ft = 0
        total_ff = 0
    
        # For each threshold, update the confusion matrix.
        # We're changing all of the counts_t[thresh] predictions to false,
        # leading to fewer true positives, and changing all the counts_f[thresh]
        # predictions to true, leading to more true negatives. By doing this
        # as a running total, we don't need to compute it completely from scratch.
        for thresh in range(num_scores):
            mcc_val[thresh] = MccBinarized.mcc(total_tt, total_tf, total_ft, total_ff)
            total_tt -= counts_t[thresh]
            total_tf -= counts_f[thresh]
            total_ft += counts_t[thresh]
            total_ff += counts_f[thresh]
    
            # Sanity check -- sum of our counts should always be number of predictions.
            assert(total_tt + total_tf + total_ft + total_ff == n)
    
        return mcc_val
    
    @staticmethod
    def sk_mcc(prediction, ground_truth):
        thresholds = 256
        ground_truth = np.array(ground_truth)
        prediction = np.array(prediction)
        mcc_val = {}
        for t in range(thresholds):
            p = np.where(prediction > t, 1.0, 0.0)
#             mcc_val[t] = matthews_corrcoef(ground_truth, p)
            mcc_val[t] = matthews_corrcoef(np.where(ground_truth>0,1.0,0.0), p)
    
        return mcc_val.values()
    
    @staticmethod
    def plot_graph(predictions, ground_truth):
        binary_mcc = MccBinarized.compute(predictions, ground_truth)
        sk_mcc = MccBinarized.sk_mcc(predictions, ground_truth)
        same = 0
        diff = []
        for i, (a,b) in enumerate(zip(binary_mcc, sk_mcc)):
            if a == b:
                same +=1
            else:
                diff.append(i)
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.plot(range(len(binary_mcc)),list(sk_mcc))
        f.add_subplot(1,2,2)
        plt.plot(range(len(binary_mcc)), binary_mcc)
        plt.show()
        
    
if __name__ == '__main__':
    m = MccBinarized()
    
    ground_truth = [0,1,1,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0]
    predictions = [0, 5, 10, 2, 18, 118, 50, 75, 20] * 2
    
    a = ImageUtils.read_image("x.png")
    b = ImageUtils.read_image("y.png")
#     c = ImageUtils.read_image("c.png")
#     b = ImageUtils.resize_image(c, tuple(np.flip(a.shape)))
    
#     mean = 50.0   # some constant
#     std = 10.0    # some constant (standard deviation)
#     noisy_img = a + np.random.normal(mean, std, a.shape)
#     n = np.clip(noisy_img, 0, 255)
#     ImageUtils.display(a,n)

    ground_truth = a.ravel()
    predictions = b.ravel()
    
    t1 = time.perf_counter()
    example_mcc = MccBinarized.compute(predictions, ground_truth)
    t2 = time.perf_counter()
#     print(f'bin mcc {round((t2-t1), 3)}')
    print(f"bin mcc {max(example_mcc)}")
    
    t1 = time.perf_counter()
    sk_mcc = MccBinarized.sk_mcc(predictions, ground_truth)
    t2 = time.perf_counter()
#     print(f'sk mcc {round((t2-t1), 3)}')
    print(f"sk mcc {max(sk_mcc)}")
    sk_val = list(sk_mcc)
#     MccBinarized.plot_graph(predictions, ground_truth)
    same = 0
    diff = []
    for i, (a,b) in enumerate(zip(example_mcc, sk_val)):
        if a == b:
            same +=1
        else:
            diff.append(i)
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.plot(range(len(example_mcc)),sk_val)
    f.add_subplot(1,2,2)
    plt.plot(range(len(example_mcc)), example_mcc)
    plt.show()
    a=1