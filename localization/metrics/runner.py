from metrics import Metrics
from img_ref_builder import ImgRefBuilder


def metric_scoring():
    data_path = "../data/metrics/"
    metrics = Metrics()
    data = metrics.read_data(data_path)
    metrics.start(data[2:3]) 

def model_scoring():
    irb = ImgRefBuilder()
    data = irb.get_img_ref_data()
    metrics = Metrics()
    metrics.start(data)

model_scoring()
