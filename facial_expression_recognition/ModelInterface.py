class ModelInterface(object):
    # required parameters
    def __init__(self):
        self._img_dim = 64
        self._depth = 3
        self._dropout = 0.25
        self._init_lr = 0.01
        self._nr_classes = 7

        # -----
        self._batch_size = 128
        self._epochs = 2
        self._split = 80

    def get_img_dim(self):
        return self._img_dim

    def get_depth(self):
        return self._depth

    def get_dropout(self):
        return self._dropout

    def get_init_lr(self):
        return self._init_lr

    def get_nr_classes(self):
        return self._nr_classes

    def get_batch_size(self):
        return self._batch_size

    def get_epochs(self):
        return self._epochs

    def get_split(self):
        return self._split

    def initialize_model(self):
        raise NotImplementedError

    def set_params(self,  img_dim, dropout, init_lr, classes_no, depth, batch_size, epochs):
        self._img_dim = img_dim
        self._depth = depth
        self._dropout = dropout
        self._init_lr = init_lr
        self._nr_classes = classes_no
        self._batch_size = batch_size
        self._epochs = epochs

    def get_name(self):
        raise NotImplementedError

    def fit(self, train_x, train_y, test_x, test_y, epochs, batch_size):
        raise NotImplementedError

    def save_weights(self,path): # for now, the model is saved as a .h5 file in every case
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def get_history(self):
        raise NotImplementedError