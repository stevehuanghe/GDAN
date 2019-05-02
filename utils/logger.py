import logging
import pprint


class Logger(object):
    def __init__(self, filename=None, level=logging.INFO):
        if filename is not None:
            logging.basicConfig(level=level, filename=filename,
                                format='%(asctime)s - %(levelname)s - %(name)s: %(message)s') # %(name)s
        else:
            logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s: - %(message)s')

        self.filename = filename

    def get_logger(self, name=__name__):
        logger = logging.getLogger(name)
        if self.filename is not None and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


def log_args(filename, args):
    with open(filename, 'a') as out:
        pprint.pprint(vars(args), stream=out)


if __name__ == '__main__':
    logger = Logger().get_logger()
    logger.info('test')




