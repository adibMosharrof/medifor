import logging

class LogUtils:
    
    @staticmethod
    def init_log(output_dir):
        logging.basicConfig(filename='{}/app.txt'.format(output_dir), level=logging.INFO, format='%(message)s')
        my_logger = logging.getLogger()
        return my_logger