
from dataset import AuxTables
from .featurizer import Featurizer

import torch
#GM
class CurrentInitFeautizer(Featurizer):
    def specific_setup(self):
        self.name = 'Current_initSignal'
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)

        self.attrs_number = len(self.ds.attr_to_idx)

    def gen_feat_tensor(self, input, classes):

        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        object = input[3]

        # 1x(max domain size) x (number of attributes)
        # sets index corresponding to value to 1 if Current_Init dataframe agrees with this value
        tensor = torch.zeros(1, classes, self.attrs_number)
        if self.iteration != 0:
            for idx, val in enumerate(domain):
                if self.current_init[attribute][object] == str(val):
                    tensor[0][idx][self.ds.attr_to_idx[attribute]] = 1

        return tensor

    def create_tensor(self):
        self.current_init = self.ds.aux_table[AuxTables.current_init].df.to_dict()

        # returns a concatenation of feature tensors for each variable
        # dimension is (# of vars) X (max domain size) X 1
        query = 'SELECT _vid_, attribute, domain, object FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
        results = self.ds.engine.execute_query(query)
        tensors = [self.gen_feat_tensor(res, self.classes) for res in
                   results]
        combined = torch.cat(tensors)
        self.tensor = combined

        return self.tensor

    def feature_names(self):
        return self.all_attrs