import logging
import time
import os

def create_logger(logger_name = "log", root_path = './logs/'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Log等级总开关

    curr_time =  time.localtime(time.time())
    log_dir = time.strftime('%Y-%m-%d', curr_time)
    log_file_name = time.strftime('%H-%M-%S', curr_time)

    save_dir = root_path + log_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_dir = save_dir + "/" + log_file_name + ".txt"
    fh = logging.FileHandler(log_file_dir, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    logger.addHandler(fh)
    return logger, save_dir

