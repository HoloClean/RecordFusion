import sys

sys.path.append('../')
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *

from repair.learn.other_algorithm import *

'''
Stage 1: Initialization of HoloFusion
a) holoclean.HoloClean creates a holoclean objects: all the flag are the same as holoclean except we need fusion=True
'''

# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_name='holo',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    test2train=0.95,
    nb_cor_strength=0.8,
    epochs=30,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    verbose=False,
    timeout=3 * 60000,
    feature_norm=False,
    weight_norm=False,
    print_fw=True,
    fusion=True,
    statistics=False,
    balance=3
).session

'''
Stage 2: Input of HoloFusion Code:
Purpose : We read the dataset and the ground truth. we also set different variables

a) add_source_name : sets attribute to be used to describe source of data 
    Input = key: the attribute that is the source
    Effect : set the source variable
b) add_ratio : Specify the ratio of testing/training data
    Input = ratio of testing/training data
    Effect : set the ratio variable
c) add key : Specify the attribute  that is the key of the cluster
    Input: key: the attribute that is the key of the cluster
    Effect: set the key variable
d) load data :  load_data takes the filepath to a CSV file to load as the initial dataset. (same as holoclean)
    Input: name: (str) name to initialize dataset with (in the database)
    Input: fpath: (str) filepath to CSV file.
    Effect : set the dataset dataframe and we also save it in the database (schema the dataset schema) (like holoclean)
e) read_ground: method to read the ground truth
    Input: fpath: (str) filepath to test set (ground truth) CSV file.
    Input: tid_col: (str) column in CSV that corresponds to the TID.
    Input: attr_col: (str) column in CSV that corresponds to the attribute.
    Input:  val_col: (str) column in CSV that corresponds to correct value for the current TID and attribute (i.e. cell).
    Input:na_values: (Any) how na_values (null) are represented in the data. (like holoclean) 
        Calls:  load_data_fusion 
            Input : the same input 
            Output: that creates the double dictionary  self.ds.correct_object_dict where the first key is the object
            (key of cluster example flight number)
            second key the attribute and the value the correct value 
            (example object_dict[object_key][attribute]= correct value)

    Effect : save the ground truth dataframe create a table in the database and creates the self.ds.correct_object_dict dictionary
    schema: _tid_ (cluster key example flight number) | _attribute_  |  _value_


'''

# GM
hc.add_source_name('src')  # sets attribute to be used to describe source of data
hc.add_ratio(0.95)  # sets ratio of testing/training data

#
hc.add_key('flight') # sets attribute to be used as key
hc.load_data('flight_sept','../testdata/flight/flight_sept/flight_input_holo.csv') # load data from CSV file
hc.read_ground('../testdata/flight/flight_sept/flaten_input_record.csv',tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data
# #


# hc.add_key('date_symbol')
# hc.load_data('stock110k','../testdata/stock/stock_110k2/stock_input_holo.csv')
# hc.read_ground('../testdata/stock/stock_110k2/flaten_input_record.csv', tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data


# hc.add_key('timeland') # sets attribute to be used as key
# hc.load_data('weather','../testdata/weather/weather_input_holo.csv') # load data from CSV file
# hc.read_ground('../testdata/weather/flaten_input_record.csv',tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data

#
# hc.add_key('isbn')  # sets attribute to be used as key
# hc.load_data('book',
#              '../testdata/book/book_input_holo.csv')  # load data from CSV file
# hc.read_ground('../testdata/book/flaten_input_record.csv', tid_col='tid',
#                attr_col='attribute',
#                val_col='correct_val')  # read ground truth data
#

# hc.add_key('ein') # sets attribute to be used as key
# hc.load_data('address','../testdata/address/address_input_holo.csv') # load data from CSV file
# hc.read_ground('../testdata/address/flaten_input_record_address.csv', tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data
# hc.load_dcs('../testdata/address/address_constraints.txt')
# hc.ds.set_constraints(hc.get_dcs())


# 3. Repair errors utilizing the defined features.

hc.setup_objects()
algorithm = 1
other_iterations = 100

recurrent_iterations = 15
hc.recurrent_iterations = recurrent_iterations

hc.ds._create_current_init_general()  # create initial guess at data using majority vote
cooccur = OccurAttrFeaturizerfusion()
row = nnFeaturizer_row()
col = nnFeaturizer_col()
freq = FreqFeaturizerFusion()
current_init = CurrentInitFeautizer()
ngram = ngramFeaturizer()

for recurrent in range(0, recurrent_iterations):
    print("iterations:" + str(recurrent))
    if recurrent == 0:
        featurizers = [cooccur, row, col, freq, current_init, ngram]
        # featurizers = [OccurAttrFeaturizerfusion(), nnFeaturizer_row(), nnFeaturizer_col(), FreqFeaturizerFusion(), CurrentInitFeautizer(),ngramFeaturizer(), DCFeaturizerFusion()]
    if recurrent == recurrent_iterations - 1:
        hc.repair_errors_fusion(featurizers, recurrent, 1)
    else:
        hc.repair_errors_fusion(featurizers, recurrent)
hc.evaluate_fusion()# measure precision of fusion algorithm
