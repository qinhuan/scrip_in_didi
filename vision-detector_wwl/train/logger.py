#!/usr/bin/env python
# coding=utf-8

import os

def logger(workspace):
    import logging
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    # 创建一个handler, 用于写入日志文件
    logger_fh = logging.FileHandler(os.path.join(workspace, 'log-root.log'))
    # 创建一个handler, 用于输出到控制台
    logger_ch = logging.StreamHandler()
    # 定义handler的输出格式formatter
    logger_formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s')
    logger_fh.setFormatter(logger_formatter)
    logger_ch.setFormatter(logger_formatter)
    logger.addHandler(logger_fh)
    logger.addHandler(logger_ch)
    return logger


