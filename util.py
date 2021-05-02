from torch import device
import argparse
import logging
import json

def batch_to_device(batch, target_device: device):
    """
    send a batch to a device

    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    moved_batch = {}
    for key, val in batch.items():
        val = val.to(target_device)
        moved_batch[key] = val
    return moved_batch

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Logger wrapper
    """
    # setup logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S',
                        filename=log_file,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    log = logging.getLogger(name)
    """
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    """

    return log

def dump_json(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile, indent=2)
    outfile.close()

