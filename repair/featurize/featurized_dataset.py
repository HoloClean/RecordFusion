from collections import namedtuple
import logging

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from domain import *


from dataset import AuxTables, CellStatus

FeatInfo = namedtuple('FeatInfo', ['name', 'size', 'learnable', 'init_weight',
                                   'feature_names'])


class FeaturizedDataset:
    def __init__(self, dataset, env):
        self.ds = dataset
        self.env = env

        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)
        self.featurizer_info = {}
        self.tensor = {}

        self.total_vars, self.classes = self.ds.get_domain_info()
        logging.debug("generating weak labels...")

        # GM
        if self.env['fusion']:
            self.weak_labels = self.generate_weak_labels_fusion()
        else:
            self.weak_labels, self.is_clean = self.generate_weak_labels()

        logging.debug("DONE generating weak labels.")
        logging.debug("generating mask...")
        self.create_augmentation()
        #update the domain based on the augmentated values
        self.total_vars, self.classes = self.ds.get_domain_info()
        self.var_class_mask, self.var_to_domsize = self.generate_var_mask()

    def create_augmentation(self):
        dfa = DFA_creation(self.env, self.ds)
        dfa.setup()
        augmentation = Augmentation(self.env, self.ds)
        augmentation.create_sampling()

    def create_features(self, featurizers, iteration_number):
        self.processes = self.env['threads']
        for f in featurizers:
            f.setup_featurizer(self.ds, self.processes, self.env['batch_size'])
            f.iteration = iteration_number
        tensors = [f.create_tensor() for f in featurizers]
        self.featurizer_info = [FeatInfo(featurizer.name,
                                         tensor.size()[2],
                                         featurizer.learnable,
                                         featurizer.init_weight,
                                         featurizer.feature_names())
                                for tensor, featurizer in zip(tensors, featurizers)]
        tensor = torch.cat(tensors, 2)
        self.tensor = F.normalize(tensor, p=2, dim=1)

    def generate_weak_labels(self):
        """
        generate_weak_labels returns a tensor where for each VID we have the
        domain index of the initial value.

        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the initial value for the i-th
            variable/VID.
        """
        # Trains with clean cells AND cells that have been weak labelled.
        query = 'SELECT _vid_, weak_label_idx, fixed, (t2._cid_ IS NULL) AS clean ' \
                'FROM {} AS t1 LEFT JOIN {} AS t2 ON t1._cid_ = t2._cid_ ' \
                'WHERE t2._cid_ is NULL ' \
                '   OR t1.fixed != {};'.format(AuxTables.cell_domain.name,
                                               AuxTables.dk_cells.name,
                                               CellStatus.NOT_SET.value)
        res = self.ds.engine.execute_query(query)
        if len(res) == 0:
            raise Exception(
                "No weak labels available. Reduce pruning threshold.")
        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        is_clean = torch.zeros(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            fixed = int(tuple[2])
            clean = int(tuple[3])
            labels[vid] = label
            is_clean[vid] = clean
        return labels, is_clean

    # GM
    def generate_weak_labels_fusion(self):
        """
        generate labels returns a tensor where for each VID we have the domain
        index of the correct value for the training data , -1 for the testing data.
        :return: Torch.Tensor of size (# of variables) X 1 where tensor[i][0]
            contains the domain index of the correct value for the i-th
            variable/VID.
        """
        self.valid_y = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        if self.env['verbose']:
            print("Generating weak labels.")
        if self.ds.x_testing is None:
            # should be a variable
            self.ds.x_training, self.ds.x_testing = train_test_split(
                self.ds.aux_table[AuxTables.entity].df,
                test_size=self.env['test2train'], random_state=self.ds.seed)
            self.ds.x_training, self.ds.x_validation = train_test_split(
                self.ds.x_training, test_size=self.ds.ratio_validation,
                random_state=self.ds.seed)

            self.ds.generate_aux_table(AuxTables.dk_cells,
                                       self.ds.x_testing, store=True)
            self.ds.generate_aux_table(AuxTables.clean_cells,
                                       self.ds.x_training, store=True)

            self.ds.generate_aux_table(AuxTables.validation_cells,
                                       self.ds.x_validation, store=True)

        query = 'SELECT _vid_, correct_value_idx FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name is not NULL;' % (
            AuxTables.cell_domain.name, AuxTables.clean_cells.name)
        res = self.ds.engine.execute_query(query)

        labels = -1 * torch.ones(self.total_vars, 1).type(torch.LongTensor)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            label = int(tuple[1])
            labels[vid] = label
        if self.env['verbose']:
            print ("DONE generating weak labels.")
        query = "SELECT _vid_, correct_value_idx FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name is not NULL ;" % (
            AuxTables.cell_domain.name, AuxTables.validation_cells.name)
        res = self.ds.engine.execute_query(query)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            labels[vid] = -3
            self.valid_y[vid] = int(tuple[1])

        return labels

    def generate_var_mask(self):
        """
        generate_var_mask returns a mask tensor where invalid domain indexes
        for a given variable/VID has value -10e6.

        An invalid domain index is possible since domain indexes are expanded
        to the maximum domain size of a given VID: e.g. if a variable A has
        10 unique values and variable B has 6 unique values, then the last
        4 domain indexes (index 6-9) of variable B are invalid.

        :return: Torch.Tensor of size (# of variables) X (max domain)
            where tensor[i][j] = 0 iff the value corresponding to domain index 'j'
            is valid for the i-th VID and tensor[i][j] = -10e6 otherwise.
        """
        var_to_domsize = {}
        query = 'SELECT _vid_, domain_size FROM %s' % AuxTables.cell_domain.name
        res = self.ds.engine.execute_query(query)
        mask = torch.zeros(self.total_vars, self.classes)
        for tuple in tqdm(res):
            vid = int(tuple[0])
            max_class = int(tuple[1])
            mask[vid, max_class:] = -10e6
            var_to_domsize[vid] = max_class
        return mask, var_to_domsize

    def get_tensor(self):
        return self.tensor

    def get_training_data(self):
        """
        get_training_data returns X_train, y_train, and mask_train
        where each row of each tensor is a variable/VID and
        y_train are weak labels for each variable i.e. they are
        set as the initial values.

        This assumes that we have a larger proportion of correct initial values
        and only a small amount of incorrect initial values which allow us
        to train to convergence.
        """
        train_idx = (self.weak_labels > -1).nonzero()[:, 0]

        X_train = self.tensor.index_select(0, train_idx)
        Y_train = self.weak_labels.index_select(0, train_idx)
        mask_train = self.var_class_mask.index_select(0, train_idx)
        return X_train, Y_train, mask_train

    def get_infer_data(self):
        """
        Retrieves the samples to be inferred i.e. DK cells.
        """
        # only infer on those that are DK cells
        infer_idx = (self.is_clean == 0).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx

    # ----- fusion
    def get_infer_data_fusion(self):
        """
        we get all the features for all the data
        :return:
        X_infer : feature tensor for the data
        mask_infer: mask for the data
        infer_idx : indexes of all the data in the feature tensor
        """
        infer_idx = (self.weak_labels != -2).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx

    def get_infer_data_fusion_testing(self):
        """
        we get the features for the testing data
         :return:
        X_infer : feature tensor for the data
        mask_infer: mask for the data
        infer_idx : indexes of the testing data in the feature tensor
        """
        infer_idx = (self.weak_labels == -1).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)
        return X_infer, mask_infer, infer_idx

    def get_infer_data_fusion_validation(self):
        """
        we get the features for the validation data
         :return:
        X_infer : feature tensor for the data
        mask_infer: mask for the data
        infer_idx : indexes of the testing data in the feature tensor
        """
        infer_idx = (self.weak_labels == -3).nonzero()[:, 0]
        X_infer = self.tensor.index_select(0, infer_idx)
        Y_train = self.valid_y.index_select(0, infer_idx)
        mask_infer = self.var_class_mask.index_select(0, infer_idx)

        return X_infer, Y_train, mask_infer, infer_idx
