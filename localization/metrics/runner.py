from metrics import Metrics

metrics = Metrics()
metrics.start() 

# exit
# #ref = np.ones((6, 6))
# size = 6
# diff = 1
# 
# sys = []
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys.append(np.arange(0., 1.0, 1.0/float(size)))
# sys = np.array(sys)*255
# cv2.imwrite('../data/metrics/00/sysMask.png',sys)