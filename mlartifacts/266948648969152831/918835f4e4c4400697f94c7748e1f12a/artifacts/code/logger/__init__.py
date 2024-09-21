import logging


class CustomLogger:
    
    def __init__(self, logger_name, log_level=logging.DEBUG):
        self.logger_name = logger_name
        self.log_level = log_level
        self.logger = self.create_logger()
        
    def create_logger(self):    
        logger = logging.getLogger(name=self.logger_name)
        file_handler = logging.FileHandler(filename='log_info.log')
        formatter = logging.Formatter(fmt="%(asctime)s-%(name)s--%(levelname)s :: %(message)s")

        # add formatter to file handler
        file_handler.setFormatter(formatter)
        # set level of file handler
        file_handler.setLevel(self.log_level)
        
        # add handler to logger
        logger.addHandler(file_handler)
        # set level of logger
        logger.setLevel(self.log_level)
        
        return logger
        
    def log_message(self,message):
        # log the message in file
        self.logger.debug(message)