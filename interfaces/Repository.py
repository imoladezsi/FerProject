class Repository(object):
    def __init__(self, algorithms):
        self.__all = algorithms

    def get_labels(self):
         return [x.get_name() for x in self.__all]

    def get_options(self):
        return self.__all