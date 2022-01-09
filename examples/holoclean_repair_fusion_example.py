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
    epochs=100,
    threads=1,
    batch_size=5,
    verbose=False,
    timeout=3 * 60000,
    weight_norm=False,
    weight_decay=0.01,
    fusion=True,
    statistics=False,
    bias=True,
    seed=45,
    optimizer='adam',
    learning_rate=0.001,
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


#GM
hc.add_source_name('src') # sets attribute to be used to describe source of data
hc.add_ratio(0.9) # sets ratio of testing/training data
hc.add_ratio_validation(0.5) # sets ratio of training/validation data


hc.add_key('flight') # sets attribute to be used as key
hc.load_data('flight_sept','../testdata/flight/flight_sept/flight_input_holo.csv') # load data from CSV file
hc.read_ground('../testdata/flight/flight_sept/flaten_input_record.csv',tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data
# #


# hc.add_key('date_symbol')
# hc.load_data('stock110k','../testdata/stock/stock_110k/stock_input_holo.csv')
# hc.read_ground('../testdata/stock/stock_110k/flaten_input_record.csv', tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data


# hc.add_key('timeland') # sets attribute to be used as key
# hc.load_data('weather','../testdata/weather/weather_input_holo.csv') # load data from CSV file
# hc.read_ground('../testdata/weather/flaten_input_record.csv',tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data

#
# hc.add_key('isbn') # sets attribute to be used as key
# hc.load_data('book','../testdata/book/book_input_holo.csv') # load data from CSV file
# hc.read_ground('../testdata/book/flaten_input_record.csv', tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data
#

# hc.add_key('ein') # sets attribute to be used as key
# hc.load_data('address','../testdata/address/address_input_holo.csv') # load data from CSV file
# hc.read_ground('../testdata/address/flaten_input_record_address.csv', tid_col='tid',attr_col='attribute', val_col='correct_val') # read ground truth data
# hc.load_dcs('../testdata/address/address_constraints.txt')
# hc.ds.set_constraints(hc.get_dcs())


# 3. Repair errors utilizing the defined features.
'''
Stage 3 : we create the random variables of the machine learning model and we create the domain for each variable
a) setup_objects : create the random variables that we will use in the machine learning model and we create the domain for each variable 
    Calls: self.create_objects.setup()
        Calls:
            self._create_object_table():Creates the domain dataframe from the initial dataset:
            Input: None
            Output : domain dataframe:
            Each row is a random variable (object + attribute) where we save the following informations:
                attribute: of the random variable
                _vid_ : id of the random variable
                domain : domain values for each random variable (example: a1 ||| a2) (the domain is create from ALL the values 
                that we see in the observations)
                domain_size: domain size of the random variable:
                correct value : which is the correct value
                correct_value_idx : the index of the correct value in the domain (we assume that the correct value is one of the values in the domain)
                object: object_key for this random variable (example flight number)
                _cid_ : cell id (we do not use it. we just have it so the cell_domain table will have the same schema with holoclean)
            
            self.store_domains(domain): Saves the domain dataframe that we got from  _create_object_table to the table cell_domain:
            schema :  _cid_ | _tid_ | _vid_ |attribute| correct_value | correct_value_idx |domain|domain_size |object 
            '''
hc.setup_objects()

"""
Stage 4 : Main HoloFusion phase: 
we do the training and the inference of our model 
choosing algorithm variable:
    1: Holofusion
    2: ACCU (needs the other_iteration variable) Data Fusion: Resolving Conflicts from Multiple Sources
    3: CATD (needs the other_iteration variable) A confidence-aware approach for truth discovery on long-taildata
    4: SLIMFAST (not implemented)
    5: Majority vote

Description of option 1 Holofusion:
recurrent_iterations variable: how many iterations we will have in HoloFusion
hc.ds._create_current_init_general() : create initial guess at data using majority vote
    Calls: self.get_statistics_fusion
    Output : 
        Calls: collect_stats_fusion
            Calls
    Ouput: Creates the current Init table where for each random variable we get the most popular value
        (we randomly choose between ties)
        
For every iteration we call the repair_errors_fusion: The main method of HoloFusion
    Input : featurizers : list of all the features
            recurrent : number of iteration
    Calls:
        a) self.repair_engine.setup_featurized_ds(featurizers, iteration_number) :
            set up and run featurizers on the dataset, store the features as a PyTorch tensor
            Input : list of all the features
            iteration number = number of current iteration
            Calls:
                FeaturizedDataset : same as holoclean except:
                    Calls:
                        generate_weak_labels_fusion (different than holoclean):
                                A)Seperates the data to training and testing using the method train_test_split and the ratio that we set before in add_ratio
                                B)generate labels returns a tensor where for each VID we
                                have the domain index of the correct value for the training data , -1 for the testing data.
            ------------------------------------------------------------------------------------------             
            Featurizers:
            all the featurizers have some basics methods:
                __init__ : Initializaiton method of the featurizers:
                
                create_tensor (like holoclean) : we iterate over the random variables we pass their domain to the gen_feat_tensor method
                
                gen_feat_tensor : populates the feature tensor for the specific random variable
                 
            
            1)SourceFeaturizer():
            Calls:
                _init_ : Initializaiton method 
                Calls: 
                    self.ds.get_statistics_sources() : 
                    Calls: 
                        self.collect_stats_sources():
                    Effect creates self.source_stats dictionary dictionary
                         For each object, attribute , value we have a list of sources that claims that this object-attribute takes this value
                         example: self.source_stats dictionary[object][attribute][val] = [source_index_1, source_index2,...]
                Effect : set the self.source_stats dictionary (describe above): 
                         set the self.number_of_sources  : number of sources
                
                gen_feat_tensor :
                    Input:
                    input : has information about vid, attribute, domain and object key for the random variable
                    classes: max_domain size
                    For each random variable creates a zero tensor with dimensions: 1 X (max domain size) X (number of sources)
                    For each possible value with index idx for each source with index source_index that claims that this possible value is the correct value
                    we get:
                                    tensor[0][idx][source_index] = 1

                    
            
            2)OccurAttrFeaturizerfusion()
                        Calls:
                _init_ : Initialization method 
                Calls: 
                    self.create_cooccur_stats_dictionary : 
                    Effect:
                        self.cooccur_pair_stats: 
                            counts frequency of two observed values occurring together for two of an entity's attributes 
                            self.cooccur_pair_stats[entity][attr1][attr2][(val1, val2)] = n
                        self.domain_stats:
                            counts frequency of observed values for a particular entity's attributes  e.g. self.domain_stats[entity][attr][val] = n
                    
                create_tensor: 
                Calls:
                    create_cooccur_dictionary
                    Effect : create a dictionary of current inferred values of dataset's cells
                    example self.dictionary_cooccur[object_key][attr] = value
                gen_feat_tensor :
                    input : has information about vid, attribute, domain and object key for the random variable
                    
                    classes: max_domain size
                    For each random variable creates a zero tensor with dimensions: 
                     dimension is 1 X (max domain size) X (number of attributes * number of attributes)
                    For each possible value with index idx we set the tensor to the value to the co-occur value that has  with the value 
                    that co-occur with in another attribute in current_init
                    tensor[0][rv_domain_idx[rv_val]][index] = prob
                    where:
                     index: rv_attr_idx * self.attrs_number + attr_idx (as in holoclean)
                     prob = self.domain_stats[object][attr][co_value] / self.cooccur_pair_stats[object][attr][rv_attr].get((co_value, rv_val), 0)
                     rv_domain_idx[rv_val] : index of the possible value in the domain 
                    
            
            3)nnFeaturizer_row()
                Calls:
                    _init_ : Initialization method 
                    Calls
                        self.create_embedding():Creates the embedding model
                            We train a word-embedding model, where the input dataset D is considered to be a document and each tuple in D is a sentence 
                            Effect: save the embedding model in self.embedding
                        self.create_average_vectors(): Creates the average embedding vectors
                            For each object and attribute we have a vector which is the average embedding vector of all the possible values
                                self.average_embeddings[object][attribute] = vector
                    
                    
                    gen_feat_tensor :
                        input : has information about vid, attribute, domain and object key for the random variable
                        classes: max_domain size
                        For each random variable creates a zero tensor with dimensions: # 1x(max domain size)x1

                        For each possible value with index idx we update the tensor to be a function  cosine distance between embedding of
                        this value and the average embedding vector:
                        count = 2 * dist.cosine(embedding_vector, self.average_embeddings[object_key][attribute]) - 1
                        if count > 0:
                            count = float(count) / 3
                         
            4)nnFeaturizer_col()
                Calls:
                    _init_ : Initialization method 
                    Calls
                        self.create_embedding():Creates the embedding models one for each attribute
                        We train a word-embedding model for each attribute, where each column in the dataset 
                        D is considered to be a document and each possible value a sentence.
                            Effect: save the embedding models in self.embedding
                        self.create_average_vectors(): Creates the average embedding vectors
                            For each object and attribute we have a vector which is the average embedding vector of all the possible values
                                self.average_embeddings[object][attribute] = vector
                    
                    gen_feat_tensor :
                        input : has information about vid, attribute, domain and object key for the random variable
                        classes: max_domain size
                        For each random variable creates a zero tensor with dimensions: # 1x(max domain size)x1

                        For each possible value with index idx we update the tensor to be a function  cosine distance between embedding of
                        this value and the average embedding vector:
                        count = 2 * dist.cosine(embedding_vector, self.average_embeddings[object_key][attribute]) - 1
                        if count > 0:
                            count = float(count) / 3
            
            5)FreqFeaturizerFusion()
                Calls:
                    _init_ : Initialization method 
                    Calls:
                        self.ds.get_statistics_fusion()
                        Calls:
                            collect_stats_fusion()
                        Effect:
                            Creates : 
                                self.single_attr_stats: 
                                for each attribute and the pair (object, val) we get it's frequency:
                                example :self.single_stats[attribute][(object, val)] = 2
                                self.object_stats: 
                                    For each object (for example for each flight) we get the total number of observations:
                                    self.object_stats[object] = 7
                    Effect : set the self.source_stats dictionary (decribe above): 
                             set the self.number_of_sources  : number of sources
                    
                    gen_feat_tensor :
                        input : has information about vid, attribute, domain and object key for the random variable
                        classes: max_domain size
                        For each random variable creates a zero tensor with dimensions: # 1x(max domain size)x1

                        For each possible value with index idx we update the tensor with 
                        corresponding to value to # of times it has been observed divided by total number of observations of any value:
                            float(self.single_stats[attribute][(object, val)]) / float(self.object_stats[object])
                        
                
            6)CurrentInitFeautizer()
                Calls:
                    _init_ : Initialization method 
                        Effect : set the number of variables 
                    
                    gen_feat_tensor :
                        input : has information about vid, attribute, domain and object key for the random variable
                        classes: max_domain size
                        For each random variable creates a zero tensor with dimensions: 1x(max domain size) x (number of attributes)

                        For each possible value with index idx we update the tensor with  1 if Current_Init value for this random variable 
                        agrees with this value

            
            7)ngramFeaturizer():
                Calls:
                    _init_ : Initialization method 
                        Calls:
                            self.build_reduced_model : build a model for all the characters (for example all letter to A, all numbers to N)
                            self.build_distribution :
                            Calls:
                                init_distribution() :
                            Effect : creates pandas_dict: for each attribute we have freq list for each n_gram
                            example pandas_dict[attribute]["freq"] = ['AA':0.27, 'AN':0.5]
                    
                    gen_feat_tensor :
                        Input : has information about vid, attribute, domain and object key for the random variable
                        classes: max_domain size
                        For each random variable creates a zero tensor with dimensions:  1x(max domain size)x1

                        Calls:
                            get_ngrams : get the n_gram of a possible value
                            Input:
                            n: n for n_gram
                            word: we want the n_gram of this word
                            

                        For each possible value with index idx we update the tensor to the value 
                        corresponding to the product of n_grams of the possible value
                        
            8)DCFeaturizerFusion() : like holoclean except we use the current_init
                        
            
            
            ------------------------------------------------------------------------------------------             

        b)repair_engine.setup_repair_model() : initialize machine learning model to repair errors
        (like holoclean)
        
        c)repair_engine.fit_repair_model(): train model on repair training data
        (like holoclean)
        Calls: 
            feat_dataset.get_training_data() : get the training data,the labels of the training data and mask
            Output:
                X_train: feature matrix of all the training dataset
                Y_train: labels of the training data
                mask_train: mask of the random variables of training data
            self.repair_model.fit_model(X_train, Y_train, mask_train) : we train our model (like holoclean)
            Input:
                X_train: feature matrix of all the training dataset
                Y_train: labels of the training data
                mask_train: mask of the random variables of training data
         
        d)repair_engine.infer_repairs_fusion() : predict values for dataset
        Calls:
            -self.feat_dataset.get_infer_data_fusion(): get  data and mask 
            Ouput: 
                X_pred : feature matrix of all the dataset
                mask_pred: mask of all the random variables
                infer_idx
            -self.repair_model.infer_values : infer values for the  data
            Input: 
                X_pred : feature matrix for the data
                mask_pred: mask for the data
            Outpur:
                Y_pred : the infered vector for the data
            -self.get_infer_dataframes_fusion : select predicted values with highest probability
            Output : 
                return inferred values dataframe with schema : vid_ |inferred_assignment |prob
                return distribution dataframe with schema :  _vid_ | distribution  (like holoclean)  
        Effect: 
            save inferred values dataframe to the database with name AuxTables.inf_values_idx with schema
            _vid_ |inferred_assignment |prob
            save distribution dataframe to the databse with name AuxTables.cell_distr:
             _vid_ | distribution      (like holoclean)                         


        e)get_inferred_values_fusion() : collate and save inferred values from SQL (in this call inferred the whole dataset)
        Effect: Creates  the table inf_values_dom with schema  _vid_ |attribute  | object |rv_value
        _vid_ = id of random variable
        attribute = attribute of random variable
        object = cluster key
        rv_value = inferred value
         
        f)get_current_init(final_iteration): 
        Updates the current init table from the predicted values (we get the predicted valued from AuxTables.inf_values_dom table)
        Input:
            final_iterations: flag 1 if we are the final iterations

        g)infer_repairs():  predict values based on testing data
        Calls:
            -get_infer_data_fusion_testing(): get testing data and mask 
            Ouput: 
                X_pred : feature matrix of testing data
                mask_pred: mask of the testing random variables
                infer_idx: indexes of all the testing data
            -self.repair_model.infer_values : infer values for the testing data
            Input: 
                X_pred : feature matrix for the testing data
                mask_pred: mask for the  testing data
            Output:
                Y_pred : the infered vector for the testing  data
            -self.get_infer_dataframes_fusion : select predicted values with highest probability
            Output : 
                return inferred values dataframe with schema : vid_ |inferred_assignment |prob
                return distribution dataframe with schema :  _vid_ | distribution  
        Effect: 
            save inferred values dataframe to the database with name AuxTables.inf_values_idx with schema
            _vid_ |inferred_assignment |prob
            save distribution dataframe to the databse with name AuxTables.cell_distr:
             _vid_ | distribution      (like holoclean)      
             
   
        h)get_inferred_values_fusion() :collate and save inferred values from SQL (in this call inferred the testing dataset dataset)
        Effect: Creates  the table inf_values_dom with schema  _vid_ |attribute  | object |rv_value
        _vid_ = id of random variable
        attribute = attribute of random variable
        object = cluster key
        rv_value = inferred value
        
    =====================================================================================================
    5) majority : 
    hc.majority() : makes predictions based on majority vote:
    Calls:
        self.ds.create_majority_general()
        Calls:
            self.ds.get_statistics_fusion()
            Calls:
                collect_stats_fusion()
            Effect:
                Creates : 
                    self.single_attr_stats: 
                    for each attribute and the pair (object, val) we get it's frequency:
                    example :self.single_stats[attribute][(object, val)] = 2
                    self.object_stats: 
                        For each object (for example for each flight) we get the total number of observations:
                        self.object_stats[object] = 7
    Effect : creates the inf_values_dom dataframe with schema  "_vid_" |"object" |"attribute",|"rv_value"
    where rv_value is the most popular value for each object_key-attribute (we choose randomly between ties)
        
    
"""
algorithm = 1
other_iterations = 100

if algorithm == 1:
    recurrent_iterations = 15
    hc.ds._create_current_init_general()  # create initial guess at data using majority vote

    for recurrent in range(0, recurrent_iterations):
        print("iterations:" + str(recurrent))
        if recurrent == 0:
            featurizers = [OccurAttrFeaturizerfusion(), nnFeaturizer_row(), nnFeaturizer_col(), FreqFeaturizerFusion(),
                           CurrentInitFeautizer(), ngramFeaturizer()]
            # featurizers = [OccurAttrFeaturizerfusion(), nnFeaturizer_row(), nnFeaturizer_col(), FreqFeaturizerFusion(), CurrentInitFeautizer(),ngramFeaturizer(), DCFeaturizerFusion()]
        if recurrent == recurrent_iterations - 1:
            hc.repair_errors_fusion(featurizers, recurrent, 1)
        else:
            hc.repair_errors_fusion(featurizers, recurrent)
        hc.evaluate_fusion_recurr()
elif algorithm == 2:
    # ACCU
    src_observations, labelled, holdout = hc.prepro_for_other()
    method = Accu(labelled, src_observations)
    iterations = other_iterations
    method.solve(iterations)
    hc.ds.create_object_truth(method, holdout)
elif algorithm == 3:
    # CATD
    src_observations, labelled, holdout = hc.prepro_for_other()
    method = CATD(labelled, src_observations, 0.05)
    iterations = other_iterations
    method.solve(iterations)
    hc.ds.create_object_truth(method, holdout)
elif algorithm == 4:
    # SlimFast
    pass
    # src_observations, labelled, holdout = hc.prepro_for_other()
    # src_features = hc.create_source_feature()
    # method = Slimfast(labelled, src_observations, src_features)
    # iterations = other_iterations
    # method.solve(iterations)
    # hc.ds.create_object_truth(method, holdout)
elif algorithm == 5:
    # majority vote
    hc.majority()
hc.ds.algorithm = algorithm

"""
Stage 5: Evaluation phase
Here we will get the accuracy of our predictions:
a)evaluate_fusion : measure precision of fusion algorithm
    Calls: eval_report_fusion() :
        prints the accuracy of our mode
        Calls: evaluate_repairs_fusion() : 
            creates variables:
                total_repair = (how many predictions we have)
                correct = (how many correct predictions we ahve
                prec = float(correct]) / float(total_repair)
        (we also have the code to get the accuracy for each attribute lines: 392-405)
        Effect: set the accuracy variable
"""

hc.evaluate_fusion()  # measure precision of fusion algorithm
