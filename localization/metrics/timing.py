import atexit
from datetime import datetime
from functools import reduce


class Timing():
    line = "=" * 40
    my_logger = None
    start = None
    
    def __init__(self, my_logger):
        self.my_logger = my_logger
        self.start = datetime.now()
        atexit.register(self.endlog)
    
    def get_output(self, elapsed):
        return '{0} \nProgram Execution Time : {1} \n{0}'.format(self.line, elapsed)

    def endlog(self):
        elapsed = datetime.now() - self.start
        output = self.get_output(str(elapsed))
        self.my_logger.info(output)
        print(output)