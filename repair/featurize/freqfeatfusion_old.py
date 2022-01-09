import torch
from dataset import AuxTables
from .featurizer import Featurizer
from functools import partial


#GM

class FreqFeaturizerFusion_old(Featurizer):
    def specific_setup(self):
        self.name = 'FreqSignal_old'
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)

        if self.tensor is None:
            self.attrs_number = len(self.ds.attr_to_idx)
            single_stats , object_stats  = self.ds.get_statistics_fusion()
            self.single_stats = {}
            self.object_stats = object_stats.to_dict()
            for attr in single_stats:
                self.single_stats[attr] = single_stats[attr].to_dict()

    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        object = input[3]

        # 1x(max domain size)x1
        # sets index corresponding to value to # of times it has been observed divided by total number of observations of any value
        tensor = torch.zeros(1, classes, 1)
        for idx, val in enumerate(domain):
            prob = float(self.single_stats[attribute][(object, val)]) / float(self.object_stats[object])
            tensor[0][idx][0] = prob

        return tensor


    def create_tensor(self):
        if self.tensor is None:
            query = 'SELECT _vid_, attribute, domain, object FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
            results = self.ds.engine.execute_query(query)
            tensors = [self.gen_feat_tensor(res, self.classes) for res in results]
            combined = torch.cat(tensors)
            self.tensor = combined
        return self.tensor


    def feature_names(self):
        return self.all_attrs
