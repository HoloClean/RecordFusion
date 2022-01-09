import torch
from dataset import AuxTables
from .featurizer import Featurizer
#GM
class SourceFeaturizer(Featurizer):

    def specific_setup(self):
        self.name = 'SourceFeaturizer'
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)
        if self.tensor is None:
            self.source_stats , self.number_of_sources  = self.ds.get_statistics_sources()


    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        object = input[3]

        # set cell@(idx1, idx2) corresponding to value, source to 1 if source corroborates value
        tensor = torch.zeros(1, classes, self.number_of_sources) # dimension 1 X (max domain size) X (number of sources)
        for idx, val in enumerate(domain):
            source_list = self.source_stats[object][attribute][val]
            for source_index in source_list:
                tensor[0][idx][source_index] = 1
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
