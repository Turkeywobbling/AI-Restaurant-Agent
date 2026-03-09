import logging
import sys
import os
from venv import logger

class log_saver:
    _instance = None

    def __init__(self, log_file: str = "log_output.log", log_level: int = logging.DEBUG, on_Note_book: bool = False):
        self.is_init = False
        self.inititalize_logger(log_file, log_level, on_Note_book)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def inititalize_logger(self, log_file: str = "log_output.log", log_level: int = logging.DEBUG, on_Note_book: bool = False):    
        if self.is_init:
            print("logutils已被初始化过了")
            return  # 已经初始化过了
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # 清除已有的handler
        self.logger.handlers.clear()

        if on_Note_book:
            notebook_handler = logging.StreamHandler(sys.stdout)
            notebook_handler.setFormatter(logging.Formatter('%(asctime)s - NOTEBOOK - %(levelname)s - %(message)s'))
            self.logger.addHandler(notebook_handler)

        #  直接写入终端（Windows和Linux/Mac通用）
        terminal_handler = logging.StreamHandler(open('terminal_output.log', 'w'))    
        terminal_handler.setFormatter(logging.Formatter('%(asctime)s - TERMINAL - %(levelname)s - %(message)s'))
        self.logger.addHandler(terminal_handler)    

        self.is_init = True

    def log_debug(self, message: str):
        if not self.is_init:
            self.inititalize_logger()

        self.logger.debug(message)    

    def log_info(self, message: str):
        if not self.is_init:
            self.inititalize_logger()

        self.logger.info(message)

    def log_error(self, message: str):
        if not self.is_init:
            self.inititalize_logger()

        self.logger.error(message)
