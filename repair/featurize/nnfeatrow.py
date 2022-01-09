import scipy.spatial.distance as dist
from gensim.models import FastText
import random
import numpy as np
import torch
from dataset import AuxTables
from .featurizer import Featurizer

#GM
class nnFeaturizer_row(Featurizer):
    def specific_setup(self):
        self.name = 'nnFeaturizer_row'
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)
        self.seed = self.ds.seed

        if self.tensor is None:
            if not self.setup_done:
                raise Exception('Featurizer %s is not properly setup.'%self.name)
            self.average_embeddings ={}
            self.embedding = self.create_embedding()
            self.create_average_vectors()


    def create_embedding(self):
        """
        This method creates the embedding model
        """

        print("building row embedding \n")
        self.attributes = self.ds.raw_data.df.keys().tolist()
        self.attributes.remove(self.ds.src)
        self.attributes.remove('_tid_')

        rows =[]
        for row in self.ds.raw_data.df.to_dict('records'):
            # make list of rows
                row_list = []
                for attribute in self.attributes:
                    if attribute != self.ds.key:
                        row_list.append(self.sanitize_string(row[attribute]))
                rows.append(row_list)
        model = FastText(rows, min_count=1,  word_ngrams=0, window= len(self.attributes),seed=self.seed, workers=1)

        return model

    def create_average_vectors(self):
        """
        We create one embedding vector per attribute-object once
        """

        observations_per_object_attribute = {}
        self.average_embeddings = {}

        # iterate through init dataset rows
        for row in self.ds.raw_data.df.to_dict('records'):

            # count number of times this entity has been observed
            if row[self.ds.key] not in self.average_embeddings:
                self.average_embeddings[row[self.ds.key]] = {}
                observations_per_object_attribute[row[self.ds.key]] = {}
            # calculate the average embedding vector for each variable
            for attribute in self.attributes:
                if attribute != self.ds.key:
                    if attribute not in  observations_per_object_attribute[row[self.ds.key]]:
                        observations_per_object_attribute[row[self.ds.key]][attribute] =0
                    if len(self.sanitize_string(row[attribute])) >0:
                        observations_per_object_attribute[row[self.ds.key]][attribute]= observations_per_object_attribute[row[self.ds.key]][attribute] + 1
                        embedding = np.round(self.embedding.wv[self.sanitize_string(row[attribute])], 4)

                        if attribute not in self.average_embeddings[row[self.ds.key]]:
                            self.average_embeddings[row[self.ds.key]][attribute] = np.round(embedding,4)
                            #self.average_embeddings[row[self.ds.key]][attribute] =  np.round(embedding / np.linalg.norm(embedding), 4)

                        else:
                            self.average_embeddings[row[self.ds.key]][attribute] = np.round(self.average_embeddings[row[self.ds.key]][attribute],4) + np.round(embedding,4)
                            # self.average_embeddings[row[self.ds.key]][attribute] = np.round(self.average_embeddings[row[self.ds.key]][attribute],4) + np.round(embedding /  np.linalg.norm(embedding),4)

        # normalize the variables's average embedding vectors by dividing them by the number of observations of that variable
        for object in sorted(self.average_embeddings):
            for attribute in sorted(self.average_embeddings[object]):
                self.average_embeddings[object][attribute] = np.round(self.average_embeddings[object][attribute],4)
                self.average_embeddings[object][attribute] = np.round(self.average_embeddings[object][attribute] / float(observations_per_object_attribute[object][attribute]),4)


        return


    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        object_key = input[3]


        # 1x(max domain size)x1
        # sets index of value to be cosine distance between embedding of this value and the average embedding vector
        tensor = torch.zeros(1, classes, 1)
        for idx, val in enumerate(domain):
            embedding_vector = self.embedding.wv[self.sanitize_string(val)]
            count = 2 * dist.cosine(embedding_vector,
                                    self.average_embeddings[object_key][
                                        attribute]) - 1
            if count > 0:
                count = float(count) / 3
            tensor[0][idx][0] = count
        return tensor

    def create_tensor(self):
        if self.tensor is None:
            query = 'SELECT _vid_, attribute, domain, object FROM %s ORDER BY _vid_' % AuxTables.cell_domain.name
            results = self.ds.engine.execute_query(query)
            tensors = [self.gen_feat_tensor(res, self.classes) for res in results]
            combined = torch.cat(tensors)
            self.tensor = combined
        return self.tensor

    def sanitize_string(self, val):
        try:
            val = val.decode("utf-8")
        except:
            pass
        val = unicode(val)
        return val


    def feature_names(self):
        return self.all_attrs
