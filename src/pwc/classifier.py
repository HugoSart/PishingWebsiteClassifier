import abc
from enum import Enum
from sklearn.model_selection import train_test_split
import numpy as np
import csv


def dec_to_nparray(dd, szformat = 'f8'):
    '''
    Convert a 'rectangular' dictionnary to numpy NdArray
    entry
        dd : dictionnary (same len of list
    retrun
        data : numpy NdArray
    '''
    names = dd.keys()
    firstKey = dd.keys()[0]
    formats = [szformat] * len(names)
    dtype = dict(names = names, formats=formats)
    values = [tuple(dd[k][0] for k in dd.keys())]
    data = np.array(values, dtype=dtype)
    for i in range(1,len(dd[firstKey])) :
        values = [tuple(dd[k][i] for k in dd.keys())]
        data_tmp = np.array(values, dtype=dtype)
        data = np.concatenate((data,data_tmp))
    return data


class Website:
    attributes = dict()

    class Tag(Enum):
        IP = "ip"
        URL_LENGTH = "url_length"
        SHORT_SERVICE = "short_service"
        SYMBOL = "symbol"
        DOUBLE_SLASH_REDIRECT = "double_slash"
        PREFIX_SUFFIX = "presu"
        SUB_DOMAIN = "sub_domain"
        SSL = "ssl"
        DOMAIN_LENGTH = "domain_length"
        FAVICON = "favicon"
        PORT = "port"
        TOKEN = "token"
        REQUEST = "request"
        URL_ANCHOR = "url_anchor"
        LINK_IN_TAB = "link_in_tab"
        SFH = "sfh"
        EMAIL = "email"
        ABNORMAL_URL = "abnormal_url"
        REDIRECT = "redirect"
        MOUSE_OVER = "mouseover"
        RIGHT_CLICK = "right_click"
        POPUP = "popup"
        IFRAME = "iframe"
        AGE = "age"
        DNS = "dnv"
        TRAFFIC = "traffic"
        RANK = "rank"
        INDEX = "index"
        REFERENCES = "references"
        REPORTS = "reports"
        RESULT = "result"

    def __init__(self, attributes):
        self.attributes = attributes


class DataSet:
    websites = []


class DataProcessor:

    full_dataset = DataSet()
    train_dataset = DataSet()
    test_dataset = DataSet()

    def __read_csv(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            count = 0
            for row in reader: # aa
                self.full_dataset.websites.append(Website(row))
                count += 1

    def __split(self):
        array = np.array(self.full_dataset.websites)
        self.train_dataset, self.test_dataset = train_test_split(array)

    def build_dataset(self, path):
        self.__read_csv(path)
        self.__split()


class Classifier(object):
    __metaclass__ = abc.ABCMeta

    train_dataset = DataSet()

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    @abs.abstractmethod
    def classify(self, website):
        return
