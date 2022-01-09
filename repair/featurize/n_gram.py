from tqdm import tqdm
import numpy as np
import torch
from dataset import AuxTables
from .featurizer import Featurizer
from collections import OrderedDict
import string
import pandas as pd
from itertools import product



def get_ngrams(word, n):
    return [word[i:i+n] for i in xrange(len(word) - (n-1))]

def build_norm_model():
    """
    build a normalized language model
    :return: a key value dictionary that is 1-1
    """
    classes = "".join([string.ascii_lowercase, "0123456789 ", string.punctuation])
    tups = [(char, char) for char in classes]
    return dict(tups)

def build_reduced_model():
    lst = []
    lst.append(("".join([string.ascii_lowercase, string.ascii_uppercase]), "A"))
    lst.append(("0123456789", "N"))
    lst.append(("-/()&$%_'.,#@!;\ ", "S"))
    lst.append((string.printable, "R"))
    return OrderedDict(lst)



#GM


class ngramFeaturizer(Featurizer):
    def specific_setup(self):
        self.prog = tqdm
        self.name="ngram"
        self.size = 2
        self.smoothing = 0.5
        self.language_model = build_reduced_model()
        #----------------------------------------------------
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)

        if self.tensor is None:
            self.cache = self.build_distribution()
            if not self.setup_done:
                raise Exception('Featurizer %s is not properly setup.'%self.name)

    def build_distribution(self):
        pandas_dict = self.init_distribution()
        i= 1
        for idx, val1 in self.prog(self.ds.raw_data.df.iterrows()):
            for attribute in self.all_attrs:
                val = self.sanitize_string(val1[attribute])
                grams = get_ngrams(val, self.size)
                keys = [self.get_key(gram) for gram in grams]
                pandas_dict[attribute].loc[keys] +=1
        for attribute in self.all_attrs:
            pandas_dict[attribute]["freq"] = pandas_dict[attribute]["freq"]/np.linalg.norm(pandas_dict[attribute]["freq"], ord=1)
        return pandas_dict

    def init_distribution(self):
        """
        Builds a vector of
        |language_model|^size and includes the smoothing factor
        :return:
        """
        pandas_dict = {}
        for attribute in self.all_attrs:
            dist = np.zeros(len(self.language_model)**self.size)
            dist = self.smoothing + dist
            vocab = self.language_model.values()
            keys = ["".join(comb) for comb in product(vocab, repeat=self.size)]
            df = pd.DataFrame({"key": keys, "freq": dist})
            df = df.set_index("key")
            pandas_dict[attribute]=df
        return pandas_dict



    def gen_feat_tensor(self, input, classes):
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        object = input[3]

        # 1x(max domain size)x1
        # sets index corresponding to the product of n_grams of the possible value
        tensor = torch.zeros(1, classes, 1)
        for idx, val in enumerate(domain):
            prob = 1
            grams = get_ngrams(val, self.size)
            try:
                keys = [self.get_key(gram) for gram in grams]
                frequencies = np.array(self.cache[attribute].loc[keys]["freq"].values)
                for freq in frequencies:
                    prob = prob * freq
                tensor[0][idx][0] = prob
            except:
                pass
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

    def get_key(self, gram):
        subkeys = [self.char_to_key(char) for char in gram]
        return "".join(subkeys)

    def char_to_key(self, char):
        for dict_key in self.language_model.keys():
            if char in dict_key:
                return self.language_model[dict_key]

    def feature_names(self):
        return self.all_attrs
