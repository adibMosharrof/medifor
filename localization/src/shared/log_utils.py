import logging
import psutil
import os

class LogUtils:
    
    @staticmethod
    def init_log(output_dir):
        logging.basicConfig(filename='{}/app.txt'.format(output_dir), level=logging.INFO, format='%(message)s')
        my_logger = logging.getLogger()
        return my_logger
    
    @staticmethod
    def print_memory_usage(text=""):
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/10**6
        print(f'{text} : {memory}')
