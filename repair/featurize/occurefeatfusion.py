from .featurizer import Featurizer
from dataset import AuxTables
from dataset.dataset import dictify
import pandas as pd
import itertools
import torch
from tqdm import tqdm
#GM
class OccurFeaturizerfusion(Featurizer):
    def specific_setup(self):
        self.name = "OccurFeaturizerfusion"
        if not self.setup_done:
            raise Exception('Featurizer %s is not properly setup.'%self.name)
        if self.tensor is None:
            self.raw_data_dict = self.ds.raw_data.df.set_index(self.ds.key).to_dict('index')
            self.all_attrs = self.ds.get_attributes()
            self.all_attrs.remove(self.ds.src)
            self.all_attrs.remove(self.ds.key)

            self.create_cooccur_stats_dictionary()

            # self.all_attrs = self.ds.get_attributes()
            self.attrs_number = len(self.ds.attr_to_idx)


    def create_tensor(self):
        # Iterate over tuples in domain
        tensors = []
        # Set tuple_id index on raw_data
        t = self.ds.aux_table[AuxTables.cell_domain]
        self.create_cooccur_dictionary()
        sorted_domain = t.df.reset_index().sort_values(by=['_vid_'])[['_tid_','object', 'attribute','_vid_','domain']]
        records = sorted_domain.to_records()
        for row in tqdm(list(records)):
            #Get tuple from raw_dataset
            feat_tensor = self.gen_feat_tensor(row)
            tensors.append(feat_tensor)
        self.tensor = torch.cat(tensors)
        return self.tensor

    def gen_feat_tensor(self, row):
        tensor = torch.zeros(1, self.classes, self.attrs_number) # dimension is 1 X (max domain size) X (number of attributes)
        rv_attr = row['attribute']
        object = row['object']
        domain = row['domain'].split('|||')
        rv_domain_idx = {val: idx for idx, val in enumerate(domain)}

        # set the index corresponding to this value to the cooccurence with another value
        for attr in self.all_attrs:
            if attr != rv_attr and attr != self.ds.key :
                attr_idx = self.ds.attr_to_idx[attr]
                co_value = self.dictionary_cooccur[object][attr]
                count1 = self.domain_stats[object][attr][co_value]
                for rv_val in rv_domain_idx:
                    count = self.cooccur_pair_stats[object][attr][rv_attr].get((co_value, rv_val), 0)
                    prob = float(count) / count1
                    tensor[0][rv_domain_idx[rv_val]][attr_idx] = prob
        return tensor



    def create_cooccur_dictionary(self):
        """
        create cooccur dictionary from the Current_init
        Dictionary of current inferred values of dataset's cells
        """

        self.dictionary_cooccur = {}

        current_init_dict = self.ds.aux_table[AuxTables.current_init].df.to_dict('index')
        for object_key in current_init_dict :
            self.dictionary_cooccur[object_key] = {}
            for attr in self.all_attrs:
                if attr != self.ds.src and attr != self.ds.key:
                    self.dictionary_cooccur[object_key][attr] = current_init_dict [object_key][attr]
        return



    def create_cooccur_stats_dictionary(self):
        """
        Creates the cooccurrence for value per objects
        """

        # counts frequency of two observed values occurring together for two of an entity's attributes
        # e.g. self.cooccur_pair_stats[entity][attr1][attr2][(val1, val2)] = n
        # where n is the number of times attr1=val1 and attr2=val2 for that entity
        self.cooccur_pair_stats = {}

        # counts frequency of observed values for a particular entity's attributes
        # e.g. self.domain_stats[entity][attr][val] = n
        # where n is the number of times attr=val for that entity
        self.domain_stats = {}

        # iterate through provided dataset
        for row in self.ds.raw_data.df.to_dict('records'):

            # if an entity is not in domain_stats object,
            # initialize dictionaries
            if row[self.ds.key] not in self.domain_stats:
                self.cooccur_pair_stats[row[self.ds.key]] = {}
                self.domain_stats[row[self.ds.key]] = {}

            # create the domain_stats for each value
            # iterate through attributes
            for co_attribute in self.all_attrs:
                if co_attribute != self.ds.key and co_attribute != "src":

                    # initialize dictionaries if attribute hasn't been initialized for this entity
                    if co_attribute not in self.domain_stats[row[self.ds.key]]:
                        self.cooccur_pair_stats[row[self.ds.key]][
                            co_attribute] = {}
                        self.domain_stats[row[self.ds.key]][co_attribute] = {}

                    value = row[co_attribute]
                    if value not in self.domain_stats[row[self.ds.key]][
                        co_attribute]:
                        self.domain_stats[row[self.ds.key]][co_attribute][
                            value] = 0.0
                    self.domain_stats[row[self.ds.key]][co_attribute][
                        value] += 1.0

                    # create the cooccur_pair_stats
                    for co_attribute1 in self.all_attrs:
                        if co_attribute1 != self.ds.key and co_attribute1 != "src" and co_attribute1!= co_attribute:
                            if co_attribute1 not in \
                                    self.cooccur_pair_stats[row[self.ds.key]][
                                        co_attribute]:
                                self.cooccur_pair_stats[row[self.ds.key]][
                                    co_attribute][co_attribute1] = {}
                            value2 = row[co_attribute1]
                            assgn_tuple = (value, value2)

                            if assgn_tuple not in \
                                    self.cooccur_pair_stats[row[self.ds.key]][
                                        co_attribute][co_attribute1]:
                                self.cooccur_pair_stats[row[self.ds.key]][
                                    co_attribute][co_attribute1][
                                    assgn_tuple] = 0.0

                            self.cooccur_pair_stats[row[self.ds.key]][
                                co_attribute][co_attribute1][
                                assgn_tuple] += 1.0

        return

    def feature_names(self):
        return self.all_attrs

