 
import logging
import logging.handlers


def CreateLogger(logger_name, file):
    # Create Logger
    logger = logging.getLogger(logger_name)
 
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
 
    logger.setLevel(logging.INFO)
 
    formatter = logging.Formatter('[%(levelname)s|%(name)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    
    # Create Handlers
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)


    file_handler = logging.FileHandler(file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(file_handler)

    return logger