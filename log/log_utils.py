# 封装的日志类

import logging
import sys
import os
from venv import logger

from log.log_saver import log_saver

class log_utils:

    @staticmethod
    def d(message: str):
        log_saver.get_instance().log_debug(message)    

    @staticmethod
    def i(message: str):
        log_saver.get_instance().log_info(message)

    @staticmethod
    def e(message: str):
        log_saver.get_instance().log_error(message)        