import logging
import os
import random

import torch
import numpy as np

from dataset import Dataset
from dcparser import Parser
from domain import DomainEngine
from detect import DetectEngine
from repair import RepairEngine
from evaluate import EvalEngine

from domain.objects import Objects
from  tqdm import  tqdm

logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
gensim_logger = logging.getLogger('gensim')
root_logger.setLevel(logging.INFO)
gensim_logger.setLevel(logging.WARNING)


# Arguments for HoloClean
arguments = [
    (('-u', '--db_user'),
        {'metavar': 'DB_USER',
         'dest': 'db_user',
         'default': 'holocleanuser',
         'type': str,
         'help': 'User for DB used to persist state.'}),
    (('-p', '--db-pwd', '--pass'),
        {'metavar': 'DB_PWD',
         'dest': 'db_pwd',
         'default': 'abcd1234',
         'type': str,
         'help': 'Password for DB used to persist state.'}),
    (('-h', '--db-host'),
        {'metavar': 'DB_HOST',
         'dest': 'db_host',
         'default': 'localhost',
         'type': str,
         'help': 'Host for DB used to persist state.'}),
    (('-d', '--db_name'),
        {'metavar': 'DB_NAME',
         'dest': 'db_name',
         'default': 'holo',
         'type': str,
         'help': 'Name of DB used to persist state.'}),
    (('-t', '--threads'),
     {'metavar': 'THREADS',
      'dest': 'threads',
      'default': 20,
      'type': int,
      'help': 'How many threads to use for parallel execution. If <= 1, then no pool workers are used.'}),
    (('-dbt', '--timeout'),
     {'metavar': 'TIMEOUT',
      'dest': 'timeout',
      'default': 60000,
      'type': int,
      'help': 'Timeout for expensive featurization queries.'}),
    (('-s', '--seed'),
     {'metavar': 'SEED',
      'dest': 'seed',
      'default': 45,
      'type': int,
      'help': 'The seed to be used for torch.'}),
    (('-l', '--learning-rate'),
     {'metavar': 'LEARNING_RATE',
      'dest': 'learning_rate',
      'default': 0.001,
      'type': float,
      'help': 'The learning rate used during training.'}),
    (('-o', '--optimizer'),
     {'metavar': 'OPTIMIZER',
      'dest': 'optimizer',
      'default': 'adam',
      'type': str,
      'help': 'Optimizer used for learning.'}),
    (('-e', '--epochs'),
     {'metavar': 'LEARNING_EPOCHS',
      'dest': 'epochs',
      'default': 20,
      'type': float,
      'help': 'Number of epochs used for training.'}),
    (('-w', '--weight_decay'),
     {'metavar': 'WEIGHT_DECAY',
      'dest':  'weight_decay',
      'default': 0.01,
      'type': float,
      'help': 'Weight decay across iterations.'}),
    (('-m', '--momentum'),
     {'metavar': 'MOMENTUM',
      'dest': 'momentum',
      'default': 0.0,
      'type': float,
      'help': 'Momentum for SGD.'}),
    (('-b', '--batch-size'),
     {'metavar': 'BATCH_SIZE',
      'dest': 'batch_size',
      'default': 1,
      'type': int,
      'help': 'The batch size during training.'}),
    (('-wlt', '--weak-label-thresh'),
     {'metavar': 'WEAK_LABEL_THRESH',
      'dest': 'weak_label_thresh',
      'default': 0.90,
      'type': float,
      'help': 'Threshold of posterior probability to assign weak labels.'}),
    (('-dt1', '--domain_thresh_1'),
     {'metavar': 'DOMAIN_THRESH_1',
      'dest': 'domain_thresh_1',
      'default': 0.1,
      'type': float,
      'help': 'Minimum co-occurrence probability threshold required for domain values in the first domain pruning stage. Between 0 and 1.'}),
    (('-dt2', '--domain-thresh-2'),
     {'metavar': 'DOMAIN_THRESH_2',
      'dest': 'domain_thresh_2',
      'default': 0,
      'type': float,
      'help': 'Threshold of posterior probability required for values to be included in the final domain in the second domain pruning stage. Between 0 and 1.'}),
    (('-md', '--max-domain'),
     {'metavar': 'MAX_DOMAIN',
      'dest': 'max_domain',
      'default': 1000000,
      'type': int,
      'help': 'Maximum number of values to include in the domain for a given cell.'}),
    (('-cs', '--cor-strength'),
     {'metavar': 'COR_STRENGTH',
      'dest': 'cor_strength',
      'default': 0.05,
      'type': float,
      'help': 'Correlation threshold (absolute) when selecting correlated attributes for domain pruning.'}),
    (('-cs', '--nb-cor-strength'),
     {'metavar': 'NB_COR_STRENGTH',
      'dest': 'nb_cor_strength',
      'default': 0.3,
      'type': float,
      'help': 'Correlation threshold for correlated attributes when using NaiveBayes estimator.'}),
    (('-fn', '--feature-norm'),
     {'metavar': 'FEATURE_NORM',
      'dest': 'feature_norm',
      'default': True,
      'type': bool,
      'help': 'Normalize the features before training.'}),
    (('-wn', '--weight_norm'),
     {'metavar': 'WEIGHT_NORM',
      'dest': 'weight_norm',
      'default': False,
      'type': bool,
      'help': 'Normalize the weights after every forward pass during training.'}),
    (('-ee', '--estimator_epochs'),
     {'metavar': 'ESTIMATOR_EPOCHS',
      'dest': 'estimator_epochs',
      'default': 3,
      'type': int,
      'help': 'Number of epochs to run the weak labelling and domain generation estimator.'}),
    (('-ebs', '--estimator_batch_size'),
     {'metavar': 'ESTIMATOR_BATCH_SIZE',
      'dest': 'estimator_batch_size',
      'default': 32,
      'type': int,
      'help': 'Size of batch used in SGD in the weak labelling and domain generation estimator.'}),
    #GM
    (('-fu', '--fusion'),
     {'metavar': 'FUSION',
      'dest': 'fusion',
      'default': False,
      'type': bool,
      'help': 'If we are in the fusion mode'}),
    (('-st', '--statistics'),
     {'metavar': 'Statistics',
      'dest': 'statistics',
      'default': False,
      'type': bool,
      'help': 'If we want to show statistics for the dataset'}),
    (('-st', '--balance'),
     {'metavar': 'balance',
      'dest': 'balance',
      'default': 2,
      'type': float,
      'help': 'balance for augmentation'}),
    (('-st', '--test2train'),
     {'metavar': 'Test2train',
      'dest': 'test2train',
      'default': 0.95,
      'type': float,
      'help': 'Test to train data ratio'}),
]

# Flags for Holoclean mode
flags = [
    (tuple(['--verbose']),
        {'default': False,
         'dest': 'verbose',
         'action': 'store_true',
         'help': 'verbose'}),
    (tuple(['--bias']),
        {'default': False,
         'dest': 'bias',
         'action': 'store_true',
         'help': 'Use bias term'}),
    (tuple(['--printfw']),
        {'default': False,
         'dest': 'print_fw',
         'action': 'store_true',
         'help': 'print the weights of featurizers'}),
    (tuple(['--debug-mode']),
        {'default': False,
         'dest': 'debug_mode',
         'action': 'store_true',
         'help': 'dump a bunch of debug information to debug\/'}),
]


class HoloClean:
    """
    Main entry point for HoloClean.
    It creates a HoloClean Data Engine
    """

    def __init__(self, **kwargs):
        """
        Constructor for Holoclean
        :param kwargs: arguments for HoloClean
        """

        # Initialize default execution arguments
        arg_defaults = {}
        for arg, opts in arguments:
            if 'directory' in arg[0]:
                arg_defaults['directory'] = opts['default']
            else:
                arg_defaults[opts['dest']] = opts['default']

        # Initialize default execution flags
        for arg, opts in flags:
            arg_defaults[opts['dest']] = opts['default']

        # check env vars
        for arg, opts in arguments:
            # if env var is set use that
            if opts["metavar"] and opts["metavar"] in os.environ.keys():
                logging.debug(
                    "Overriding {} with env varible {} set to {}".format(
                        opts['dest'],
                        opts["metavar"],
                        os.environ[opts["metavar"]])
                )
                arg_defaults[opts['dest']] = os.environ[opts["metavar"]]

        # Override defaults with manual flags
        for key in kwargs:
            arg_defaults[key] = kwargs[key]

        # Initialize additional arguments
        for (arg, default) in arg_defaults.items():
            setattr(self, arg, kwargs.get(arg, default))

        # Init empty session collection
        self.session = Session(arg_defaults)


class Session:
    """
    Session class controls the entire pipeline of HC
    """

    def __init__(self, env, name="session"):
        """
        Constructor for Holoclean session
        :param env: Holoclean environment
        :param name: Name for the Holoclean session
        """
        # use DEBUG logging level if verbose enabled
        if env['verbose']:
            root_logger.setLevel(logging.DEBUG)
            gensim_logger.setLevel(logging.DEBUG)

        logging.debug('initiating session with parameters: %s', env)

        # Initialize random seeds.
        random.seed(env['seed'])
        torch.manual_seed(env['seed'])
        np.random.seed(seed=env['seed'])

        # Initialize members
        self.name = name
        self.env = env
        self.ds = Dataset(name, env)
        self.dc_parser = Parser(env, self.ds)
        self.domain_engine = DomainEngine(env, self.ds)
        self.detect_engine = DetectEngine(env, self.ds)
        self.repair_engine = RepairEngine(env, self.ds)
        self.eval_engine = EvalEngine(env, self.ds)


        # GM
        self.create_objects = Objects(env, self.ds)
        self.ds.seed = self.env['seed']


    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None):
        """
        load_data takes the filepath to a CSV file to load as the initial dataset.

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV file.
        :param na_values: (str) value that identifies a NULL value
        :param entity_col: (st) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks
            specifies the column containing the source for each "mention" of an
            entity.
        """
        status, load_time = self.ds.load_data(name,
                                              fpath,
                                              na_values=na_values,
                                              entity_col=entity_col,
                                              src_col=src_col)
        logging.info(status)
        logging.debug('Time to load dataset: %.2f secs', load_time)

    def load_dcs(self, fpath):
        """
        load_dcs ingests the Denial Constraints for initialized dataset.

        :param fpath: filepath to TXT file where each line contains one denial constraint.
        """
        status, load_time = self.dc_parser.load_denial_constraints(fpath)
        logging.info(status)
        logging.debug('Time to load dirty data: %.2f secs', load_time)

    def get_dcs(self):
        return self.dc_parser.get_dcs()

    def detect_errors(self, detect_list):
        status, detect_time = self.detect_engine.detect_errors(detect_list)
        logging.info(status)
        logging.debug('Time to detect errors: %.2f secs', detect_time)

    def setup_domain(self):
        status, domain_time = self.domain_engine.setup()
        logging.info(status)
        logging.debug('Time to setup the domain: %.2f secs', domain_time)

    def repair_errors(self, featurizers):
        status, feat_time = self.repair_engine.setup_featurized_ds(featurizers)
        logging.info(status)
        logging.debug('Time to featurize data: %.2f secs', feat_time)
        status, setup_time = self.repair_engine.setup_repair_model()
        logging.info(status)
        logging.debug('Time to setup repair model: %.2f secs', feat_time)
        status, fit_time = self.repair_engine.fit_repair_model()
        logging.info(status)
        logging.debug('Time to fit repair model: %.2f secs', fit_time)
        status, infer_time = self.repair_engine.infer_repairs()
        logging.info(status)
        logging.debug('Time to infer correct cell values: %.2f secs', infer_time)
        status, time = self.ds.get_inferred_values()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs', time)
        status, time = self.ds.get_repaired_dataset()
        logging.info(status)
        logging.debug('Time to store repaired dataset: %.2f secs', time)
        if self.env['print_fw']:
            status, time = self.repair_engine.get_featurizer_weights()
            logging.info(status)
            logging.debug('Time to store featurizer weights: %.2f secs', time)
            return status

    def evaluate(self, fpath, tid_col, attr_col, val_col, na_values=None):
        """
        evaluate generates an evaluation report with metrics (e.g. precision,
        recall) given a test set.

        :param fpath: (str) filepath to test set (ground truth) CSV file.
        :param tid_col: (str) column in CSV that corresponds to the TID.
        :param attr_col: (str) column in CSV that corresponds to the attribute.
        :param val_col: (str) column in CSV that corresponds to correct value
            for the current TID and attribute (i.e. cell).
        :param na_values: (Any) how na_values are represented in the data.
        """
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data(name, fpath, tid_col, attr_col, val_col, na_values=na_values)
        logging.info(status)
        logging.debug('Time to evaluate repairs: %.2f secs', load_time)
        status, report_time, report_list = self.eval_engine.eval_report()
        logging.info(status)
        logging.debug('Time to generate report: %.2f secs', report_time)
        return report_list

    # GM
    def setup_objects(self):
        status, object_time = self.create_objects.setup()
        logging.info(status)
        logging.debug('Time to setup the domain and objects: %.2f secs'%object_time)

    # GM
    def repair_errors_fusion(self, featurizers, iteration_number,  final_iteration=0):

        # set up and run featurizers on the dataset, store the features as a PyTorch tensor
        status, feat_time = self.repair_engine.setup_featurized_ds(featurizers, iteration_number)
        logging.info(status)
        logging.debug('Time to featurize data: %.2f secs'%feat_time)

        # initialize machine learning model to repair errors
        status, setup_time = self.repair_engine.setup_repair_model()
        logging.info(status)
        logging.debug('Time to setup repair model: %.2f secs' % feat_time)

        # train model on repair training data
        status, fit_time = self.repair_engine.fit_repair_model()
        logging.info(status)
        logging.debug('Time to fit repair model: %.2f secs'%fit_time)

        # --------------------validation--------------------
        status, infer_time = self.repair_engine.infer_repairs_fusion_validation()
        logging.info(status)
        logging.debug(
            'Time to infer correct cell values: %.2f secs' % infer_time)

        # collate and save inferred values from SQL into Dataset object
        status, time = self.ds.get_inferred_values_fusion()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs' % time)
        validation = 1
        self.evaluate_fusion_recurr(validation)
        # -----------------------------------------------------------------------

        # predict values for the dataset
        status, infer_time = self.repair_engine.infer_repairs_fusion()
        logging.info(status)
        logging.debug('Time to infer correct cell values of datase: %.2f secs'%infer_time)

        # collate and save inferred values from SQL into Dataset object
        status, time = self.ds.get_inferred_values_fusion()
        logging.info(status)
        logging.debug('Time to collect inferred values of all cells: %.2f secs' % time)

        # create Current_Init dataframe with format: Object, Attribute, Inferred Value
        status, time = self.ds.get_current_init(final_iteration)
        logging.info(status)
        logging.debug('Time to update the current init: %.2f secs' % time)

        # predict values based on test data
        status, infer_time = self.repair_engine.infer_repairs()
        logging.info(status)
        logging.debug('Time to infer correct cell values: %.2f secs'%infer_time)

        # collate and save inferred values from SQL into Dataset object
        status, time = self.ds.get_inferred_values_fusion()
        logging.info(status)
        logging.debug('Time to collect inferred values: %.2f secs' % time)


    # GM
    def add_key(self, key):
        """
        Specify the attribute  that is the key of the cluster

        :param key: the attribute that is the key of the cluster

        """
        self.ds.key = key

    # GM
    def add_source_name(self, source_name):
        """
        Specify the attribute  that describes the sources

        :param key: the attribute that is the source

        """

        self.ds.src = source_name

    # GM
    def add_ratio(self, ratio):
        """
        Specify the ratio of testing/training data

        :param ratio: ratio of testing/training data
        """
        self.env['test2train'] = ratio

    def add_balance(self, balance):
        """
        Specify the ratio of augmentation

        :param ratio: ratio of augmentation
        """

        self.ds.balance = balance -1

    def add_ratio_validation(self, ratio):
        """
        Specify the ratio of validation/training data

        :param ratio: ratio of validation/training data
        """

        self.ds.ratio_validation = ratio

    # GM
    def read_ground(self, fpath, tid_col, attr_col, val_col, na_values=None):
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data_fusion(name, fpath, tid_col, attr_col, val_col, na_values=na_values)
        logging.info(status)
        logging.debug('Time to read ground_truth: %.2f secs'%load_time)

    # GM
    def evaluate_fusion(self ):
        # print precision of HoloClean fusion algorithm
        status, report_time = self.eval_engine.eval_report_fusion()
        print(status)
        if self.env['verbose']:
            print('Time to generate report: %.2f secs' % report_time)

    def evaluate_fusion_recurr(self, validation=0 ):
        # print precision of HoloClean fusion algorithm
        status, report_time = self.eval_engine.eval_report_fusion_recurr(validation)
        print(status)
        if self.env['verbose']:
            print('Time to generate report: %.2f secs' % report_time)


    def evaluate(self, fpath, tid_col, attr_col, val_col, na_values=None):
        """
        evaluate generates an evaluation report with metrics (e.g. precision,
        recall) given a test set.

        :param fpath: (str) filepath to test set (ground truth) CSV file.
        :param tid_col: (str) column in CSV that corresponds to the TID.
        :param attr_col: (str) column in CSV that corresponds to the attribute.
        :param val_col: (str) column in CSV that corresponds to correct value
            for the current TID and attribute (i.e. cell).
        :param na_values: (Any) how na_values are represented in the data.
        """
        name = self.ds.raw_data.name + '_clean'
        status, load_time = self.eval_engine.load_data(name, fpath, tid_col, attr_col, val_col, na_values=na_values)
        logging.info(status)
        logging.debug('Time to evaluate repairs: %.2f secs', load_time)
        status, report_time, report_list = self.eval_engine.eval_report()
        logging.info(status)
        logging.debug('Time to generate report: %.2f secs', report_time)
        return report_list

    #GM
    def majority(self):
        self.ds.create_majority_general()

    def prepro_for_other(self):
        """
        Creating the input for accu, catd, slimfast
        :param training_data_path: path to ground truth
        :param ratio: ratio of training data

        :return src_observations: source observations for each cluster
        :return labelled: training data
        :return holdout:  testing data
        """

        src_observations = {}

        records = self.ds.raw_data.df.to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            if row[self.ds.src] not in src_observations:
                src_observations[row[self.ds.src]] = {}
            for attribute in self.ds.attr_to_idx:
                if attribute != self.ds.src:
                    vid = row[self.ds.key] + "+_+" + attribute
                    if row[attribute] != "_nan_" and len(row[attribute])>0:
                        src_observations[row['src']][vid] = row[attribute]

        labelled, holdout = self.ds.create_training()
        return src_observations, labelled, holdout

