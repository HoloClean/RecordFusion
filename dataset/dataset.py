from enum import Enum
import logging
import os
import time

import pandas as pd

from .dbengine import DBengine
from .table import Table, Source
from utils import dictify_df
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class AuxTables(Enum):
    c_cells = 1
    dk_cells = 2
    cell_domain = 3
    pos_values = 4
    cell_distr = 5
    inf_values_idx = 6
    inf_values_dom = 7

    # -----fusion tables
    current_init = 8
    entity = 9
    clean_cells = 10
    validation_cells = 11


class CellStatus(Enum):
    NOT_SET = 0
    WEAK_LABEL = 1
    SINGLE_VALUE = 2


def dictify(frame):
    """
    dictify converts a frame with columns

      col1    | col2    | .... | coln   | value
      ...
    to a dictionary that maps values valX from colX

    { val1 -> { val2 -> { ... { valn -> value } } } }
    """
    d = {}
    for row in frame.values:
        here = d
        for elem in row[:-2]:
            if elem not in here:
                here[elem] = {}
            here = here[elem]
        here[row[-2]] = row[-1]
    return d


class Dataset:
    """
    This class keeps all dataframes and tables for a HC session.
    """

    def __init__(self, name, env):
        self.id = name
        self.raw_data = None
        self.repaired_data = None
        self.constraints = None
        self.aux_table = {}
        for tab in AuxTables:
            self.aux_table[tab] = None
        # start dbengine
        self.engine = DBengine(
            env['db_user'],
            env['db_pwd'],
            env['db_name'],
            env['db_host'],
            pool_size=env['threads'],
            timeout=env['timeout']
        )
        # members to convert (tuple_id, attribute) to cell_id
        self.attr_to_idx = {}
        self.attr_count = 0
        # dataset statistics
        self.stats_ready = False
        # Total tuples
        self.total_tuples = 0
        # Domain stats for single attributes
        self.single_attr_stats = {}
        # Domain stats for attribute pairs
        self.pair_attr_stats = {}
        # -------fusion
        # GM
        self.x_testing = None
        self.fusion_flag = env['fusion']
        self.env = env

    # TODO(richardwu): load more than just CSV files
    def load_data(self, name, fpath, na_values=None, entity_col=None, src_col=None):
        """
        load_data takes a CSV file of the initial data, adds tuple IDs (_tid_)
        to each row to uniquely identify an 'entity', and generates unique
        index numbers for each attribute/column.

        Creates a table with the user supplied 'name' parameter (e.g. 'hospital').

        :param name: (str) name to initialize dataset with.
        :param fpath: (str) filepath to CSV file.
        :param na_values: (str) value that identifies a NULL value
        :param entity_col: (str) column containing the unique
            identifier/ID of an entity.  For fusion tasks, rows with
            the same ID will be fused together in the output.
            If None, assumes every row is a unique entity.
        :param src_col: (str) if not None, for fusion tasks
            specifies the column containing the source for each "mention" of an
            entity.
        """
        tic = time.clock()
        try:
            # Do not include TID and source column as trainable attributes
            exclude_attr_cols = ['_tid_']
            if src_col is not None:
                exclude_attr_cols.append(src_col)

            # Load raw CSV file/data into a Postgres table 'name' (param).
            self.raw_data = Table(name, Source.FILE, na_values=na_values, exclude_attr_cols=exclude_attr_cols,
                                  fpath=fpath)

            df = self.raw_data.df
            # Add _tid_ column to dataset that uniquely identifies an entity.
            # If entity_col is not supplied, use auto-incrementing values.
            # Otherwise we use the entity values directly as _tid_'s.
            if entity_col is None:
                # auto-increment
                df.insert(0, '_tid_', range(0, len(df)))
            else:
                # use entity IDs as _tid_'s directly
                df.rename({entity_col: '_tid_'}, axis='columns', inplace=True)

            # Use '_nan_' to represent NULL values
            df.fillna('_nan_', inplace=True)

            logging.info("Loaded %d rows with %d cells", self.raw_data.df.shape[0],
                         self.raw_data.df.shape[0] * self.raw_data.df.shape[1])

            # Call to store to database
            self.raw_data.store_to_db(self.engine.engine)
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))

            # Generate indexes on attribute columns for faster queries
            for attr in self.raw_data.get_attributes():
                # Generate index on attribute
                self.raw_data.create_db_index(self.engine, [attr])

            # Create attr_to_idx dictionary (assign unique index for each attribute)
            # and attr_count (total # of attributes)
            if self.fusion_flag:
                # GM
                tmp_attr_list = self.raw_data.get_attributes()
                tmp_attr_list.remove(self.src)
                tmp_attr_list.remove(self.key)
                for idx, attr in enumerate(tmp_attr_list):
                    # Map attribute to index
                    self.attr_to_idx[attr] = idx
            else:
                self.attr_to_idx = {attr: idx for idx, attr in enumerate(self.raw_data.get_attributes())}
            self.attr_count = len(self.attr_to_idx)
        except Exception:
            logging.error('loading data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def set_constraints(self, constraints):
        self.constraints = constraints

    def generate_aux_table(self, aux_table, df, store=False, index_attrs=False):
        """
        generate_aux_table writes/overwrites the auxiliary table specified by
        'aux_table'.

        It does:
          1. stores/replaces the specified aux_table into Postgres (store=True), AND/OR
          2. sets an index on the aux_table's internal Pandas DataFrame (index_attrs=[<columns>]), AND/OR
          3. creates Postgres indexes for aux_table (store=True and index_attrs=[<columns>])

        :param aux_table: (AuxTable) auxiliary table to generate
        :param df: (DataFrame) dataframe to memoize/store for this auxiliary table
        :param store: (bool) if true, creates/replaces Postgres table for this auxiliary table
        :param index_attrs: (list[str]) list of attributes to create indexes on. If store is true,
        also creates indexes on Postgres table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.DF, df=df)
            if store:
                self.aux_table[aux_table].store_to_db(self.engine.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
            if store and index_attrs:
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def generate_aux_table_sql(self, aux_table, query, index_attrs=False):
        """
        :param aux_table: (AuxTable) auxiliary table to generate
        :param query: (str) SQL query whose result is used for generating the auxiliary table.
        """
        try:
            self.aux_table[aux_table] = Table(aux_table.name, Source.SQL, table_query=query, db_engine=self.engine)
            if index_attrs:
                self.aux_table[aux_table].create_df_index(index_attrs)
                self.aux_table[aux_table].create_db_index(self.engine, index_attrs)
        except Exception:
            logging.error('generating aux_table %s', aux_table.name)
            raise

    def get_raw_data(self):
        """
        get_raw_data returns a pandas.DataFrame containing the raw data as it was initially loaded.
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.df

    def get_attributes(self):
        """
        get_attributes return the trainable/learnable attributes (i.e. exclude meta
        columns like _tid_).
        """
        if self.raw_data is None:
            raise Exception('ERROR No dataset loaded')
        return self.raw_data.get_attributes()

    def get_cell_id(self, tuple_id, attr_name):
        """
        get_cell_id returns cell ID: a unique ID for every cell.

        Cell ID: _tid_ * (# of attributes) + attr_idx
        """
        vid = tuple_id * self.attr_count + self.attr_to_idx[attr_name]
        return vid

    def get_statistics(self):
        """
        get_statistics returns:
            1. self.total_tuples (total # of tuples)
            2. self.single_attr_stats ({ attribute -> { value -> count } })
              the frequency (# of entities) of a given attribute-value
            3. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
              the statistics for each pair of attributes, attr1 and attr2, where:
                <attr1>: first attribute
                <attr2>: second attribute
                <val1>: all values of <attr1>
                <val2>: values of <attr2> that appear at least once with <val1>.
                <count>: frequency (# of entities) where attr1=val1 AND attr2=val2
        """
        if not self.stats_ready:
            logging.debug('computing frequency and co-occurrence statistics from raw data...')
            tic = time.clock()
            self.collect_stats()
            logging.debug('DONE computing statistics in %.2fs', time.clock() - tic)

        stats = (self.total_tuples, self.single_attr_stats, self.pair_attr_stats)
        self.stats_ready = True
        return stats

    def collect_stats(self):
        """
        collect_stats memoizes:
          1. self.single_attr_stats ({ attribute -> { value -> count } })
            the frequency (# of entities) of a given attribute-value
          2. self.pair_attr_stats ({ attr1 -> { attr2 -> {val1 -> {val2 -> count } } } })
            where DataFrame contains 3 columns:
              <attr1>: all possible values for attr1 ('val1')
              <attr2>: all values for attr2 that appeared at least once with <val1> ('val2')
              <count>: frequency (# of entities) where attr1: val1 AND attr2: val2
            Also known as co-occurrence count.
        """
        logging.debug("Collecting single/pair-wise statistics...")
        self.total_tuples = self.get_raw_data().shape[0]
        # Single attribute-value frequency.
        for attr in self.get_attributes():
            self.single_attr_stats[attr] = self.get_stats_single(attr)
        # Compute co-occurrence frequencies.
        for cond_attr in self.get_attributes():
            self.pair_attr_stats[cond_attr] = {}
            for trg_attr in self.get_attributes():
                if trg_attr != cond_attr:
                    self.pair_attr_stats[cond_attr][trg_attr] = self.get_stats_pair(cond_attr, trg_attr)

    def get_stats_single(self, attr):
        """
        Returns a dictionary where the keys are domain values for :param attr: and
        the values contain the frequency count of that value for this attribute.
        """
        # need to decode values into unicode strings since we do lookups via
        # unicode strings from Postgres
        return self.get_raw_data()[[attr]].groupby([attr]).size().to_dict()

    def get_stats_pair(self, first_attr, second_attr):
        """
        Returns a dictionary {first_val -> {second_val -> count } } where:
            <first_val>: all possible values for first_attr
            <second_val>: all values for second_attr that appear at least once with <first_val>
            <count>: frequency (# of entities) where first_attr=<first_val> AND second_attr=<second_val>
        """
        tmp_df = self.get_raw_data()[[first_attr, second_attr]].groupby([first_attr, second_attr]).size().reset_index(
            name="count")
        return dictify_df(tmp_df)

    def get_domain_info(self):
        """
        Returns (number of random variables, count of distinct values across all attributes).
        """
        query = 'SELECT count(_vid_), max(domain_size) FROM %s' % AuxTables.cell_domain.name
        res = self.engine.execute_query(query)
        total_vars = int(res[0][0])
        classes = int(res[0][1])
        return total_vars, classes

    def get_inferred_values(self):
        tic = time.clock()
        # index into domain with inferred_val_idx + 1 since SQL arrays begin at index 1.
        query = "SELECT t1._tid_, t1.attribute, domain[inferred_val_idx + 1] as rv_value " \
                "FROM " \
                "(SELECT _tid_, attribute, " \
                "_vid_, init_value, string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\') as domain " \
                "FROM %s) as t1, %s as t2 " \
                "WHERE t1._vid_ = t2._vid_" % (AuxTables.cell_domain.name, AuxTables.inf_values_idx.name)
        self.generate_aux_table_sql(AuxTables.inf_values_dom, query, index_attrs=['_tid_'])
        self.aux_table[AuxTables.inf_values_dom].create_db_index(self.engine, ['attribute'])
        status = "DONE collecting the inferred values."
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

    def get_repaired_dataset(self):
        tic = time.clock()
        init_records = self.raw_data.df.sort_values(['_tid_']).to_records(index=False)
        t = self.aux_table[AuxTables.inf_values_dom]
        repaired_vals = dictify_df(t.df.reset_index())
        for tid in repaired_vals:
            for attr in repaired_vals[tid]:
                init_records[tid][attr] = repaired_vals[tid][attr]
        repaired_df = pd.DataFrame.from_records(init_records)
        name = self.raw_data.name + '_repaired'
        self.repaired_data = Table(name, Source.DF, df=repaired_df)
        self.repaired_data.store_to_db(self.engine.engine)
        status = "DONE generating repaired dataset"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

    # -------fusion metods----------
    # GM

    def get_stats_object_fusion(self):
        tmp_df = self.get_raw_data()[[self.key]].groupby([self.key]).size()
        return tmp_df

        # GM

    def get_stats_single_fusion(self, attr):
        tmp_df = self.get_raw_data()[[self.key, attr]].groupby([self.key, attr]).size()
        return tmp_df

        # GM

    def collect_stats_fusion(self):
        self.total_tuples = self.get_raw_data().shape[0]
        self.object_stats = self.get_stats_object_fusion()
        for attr in self.get_attributes():
            if attr != self.key:
                self.single_attr_stats[attr] = self.get_stats_single_fusion(attr)

        # GM

    def get_statistics_fusion(self):
        if not self.stats_ready:
            self.collect_stats_fusion()
        stats = (self.single_attr_stats, self.object_stats)
        return stats

        # GM

    def _create_current_init(self):
        """
        create a new current init
        """
        single_stats, object_stats = self.get_statistics_fusion()
        self.single_stats = {}
        for attr in single_stats:
            self.single_stats[attr] = single_stats[attr].to_dict()

        majority_vote = {}
        majority_vote_freq = {}
        majority_vote[self.key] = {}
        for attr in single_stats:
            if attr != self.src:
                for pair, freq in single_stats[attr].iteritems():
                    object = pair[0]
                    value = pair[1]
                    if value != "_nan_":
                        if attr not in majority_vote:
                            majority_vote[attr] = {}
                            majority_vote_freq[attr] = {}
                        if object not in majority_vote[attr]:
                            majority_vote[attr][object] = value
                            majority_vote_freq[attr][object] = freq
                        if freq > majority_vote_freq[attr][object]:
                            majority_vote_freq[attr][object] = freq
                            majority_vote[attr][object] = value
                        if object not in majority_vote[self.key]:
                            majority_vote[self.key][object] = object
        pd1 = pd.DataFrame.from_dict(majority_vote)
        self.generate_aux_table(AuxTables.current_init, pd1, store=True)
        return

    def _create_current_init_general(self):
        """
        create a new current init with majority with random selection in ties
        """
        # create Current_Init dataframe with format: Object, Attribute, Inferred Value
        single_stats, object_stats = self.get_statistics_fusion()
        self.single_stats = {}
        for attr in single_stats:
            self.single_stats[attr] = single_stats[attr].to_dict()

        majority_vote = {}
        majority_vote_freq = {}
        majority_vote[self.key] = {}
        for attr in single_stats:
            if attr != self.src:
                for pair, freq in single_stats[attr].iteritems():
                    object = pair[0]
                    value = pair[1]
                    if value != "_nan_":
                        if attr not in majority_vote:
                            majority_vote[attr] = {}
                            majority_vote_freq[attr] = {}
                        if object not in majority_vote[attr]:
                            majority_vote[attr][object] = [value]
                            majority_vote_freq[attr][object] = freq
                        if freq > majority_vote_freq[attr][object]:
                            majority_vote_freq[attr][object] = freq
                            majority_vote[attr][object] = [value]
                        elif freq == majority_vote_freq[attr][object]:
                            majority_vote[attr][object].append(value)
                        if object not in majority_vote[self.key]:
                            majority_vote[self.key][object] = [object]
        for attr in majority_vote:
            for object in majority_vote[attr]:
                if len(majority_vote[attr][object]) > 1:
                    value = random.choice(majority_vote[attr][object])
                    majority_vote[attr][object] = value
                else:
                    majority_vote[attr][object] = majority_vote[attr][object][0]

        pd1 = pd.DataFrame.from_dict(majority_vote)
        self.generate_aux_table(AuxTables.current_init, pd1, store=True)
        return

        # GM

    def get_current_init(self, final_iteration):
        tic = time.clock()
        try:
            init_records = self.aux_table[AuxTables.current_init].df.to_dict('index')
            t = self.aux_table[AuxTables.inf_values_dom]
            repaired_vals = dictify(t.df.reset_index())
            for tid in repaired_vals:
                for obj in repaired_vals[tid]:
                    for attr in repaired_vals[tid][obj]:
                        init_records[obj][attr] = repaired_vals[tid][obj][attr]
            repaired_df = pd.DataFrame.from_dict(init_records, orient='index')
            self.generate_aux_table(AuxTables.current_init, repaired_df, store=True)
            status = "DONE generating current init dataset"
            if final_iteration:
                name = self.raw_data.name + '_repaired'
                self.repaired_data = Table(name, Source.DF, df=repaired_df)
                self.repaired_data.store_to_db(self.engine.engine)
                status = "DONE generating repaired dataset"
        except Exception as e:
            status = "ERROR when generating repaired dataset: %s"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

        # GM

    def get_inferred_values_fusion(self):
        tic = time.clock()

        # create new SQL table with format: Object ID, Object Name, Attribute, Inferred Value
        query = "SELECT t1._tid_, t1.object, t1.attribute, domain[inferred_assignment + 1] as rv_value " \
                "FROM " \
                "(SELECT _tid_, object, attribute, " \
                "_vid_, string_to_array(regexp_replace(domain, \'[{\"\"}]\', \'\', \'gi\'), \'|||\') as domain " \
                "FROM %s) as t1, %s as t2 " \
                "WHERE t1._vid_ = t2._vid_" % (AuxTables.cell_domain.name, AuxTables.inf_values_idx.name)
        try:
            self.generate_aux_table_sql(AuxTables.inf_values_dom, query,
                                        index_attrs=['_tid_'])  # save SQL table into Dataset object
            self.aux_table[AuxTables.inf_values_dom].create_db_index(self.engine, ['attribute'])
            status = "DONE colleting the inferred values."
        except Exception as e:
            status = "ERROR when colleting the inferred values: %s" % str(e)
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

        # GM

    def get_repaired_dataset_fusion(self):
        tic = time.clock()
        try:
            init_records = {}
            t = self.aux_table[AuxTables.inf_values_dom]
            repaired_vals = dictify(t.df.reset_index())
            for tid in repaired_vals:
                for obj in repaired_vals[tid]:
                    for attr in repaired_vals[tid][obj]:
                        if tid not in init_records:
                            init_records[tid] = {}
                        try:
                            init_records[tid][attr] = repaired_vals[tid][obj][attr]
                        except:
                            pass
            repaired_df = pd.DataFrame.from_dict(init_records, orient='index')
            name = self.raw_data.name + '_repaired'
            self.repaired_data = Table(name, Source.DF, repaired_df)
            self.repaired_data.store_to_db(self.engine.engine)
            status = "DONE generating repaired dataset"
        except Exception as e:
            status = "ERROR when generating repaired dataset: %s"
        toc = time.clock()
        total_time = toc - tic
        return status, total_time

        # GM

    def get_statistics_sources(self):
        if not self.stats_ready:
            value_to_source, number_of_sources = self.collect_stats_sources()
        return (value_to_source, number_of_sources)

        # GM

    def collect_stats_sources(self):
        sources_index = {}
        value_to_source = {}
        attributes = self.raw_data.df.keys().tolist()
        attributes.remove(self.src)
        attributes.remove('_tid_')
        attributes.remove(self.key)

        for row in self.raw_data.df.to_dict('records'):
            if row[self.src] not in sources_index:
                sources_index[row[self.src]] = len(sources_index)
            if row[self.key] not in value_to_source:
                value_to_source[row[self.key]] = {}

            for attribute in attributes:
                if attribute not in value_to_source[row[self.key]]:
                    value_to_source[row[self.key]][attribute] = {}
                if row[attribute] not in value_to_source[row[self.key]][attribute]:
                    value_to_source[row[self.key]][attribute][row[attribute]] = []
                value_to_source[row[self.key]][attribute][row[attribute]].append(sources_index[row[self.src]])

        return (value_to_source, len(sources_index))

    def create_majority_general(self):
        """
        majority vote that when we have tie we choose randomly from the values
        """
        majority_dict = {}
        raw_data = {}
        raw_data['_vid_'] = []
        raw_data['object'] = []
        raw_data['attribute'] = []
        raw_data['rv_value'] = []
        x_training, x_testing = train_test_split(self.aux_table[AuxTables.entity].df, test_size=self.env['test2train'],
                                                 random_state=self.seed)
        self.generate_aux_table(AuxTables.dk_cells, x_testing, store=True)

        self.attrs_number = len(self.attr_to_idx)
        single_stats, object_stats = self.get_statistics_fusion()
        self.single_stats = {}
        self.object_stats = object_stats.to_dict()
        for attr in single_stats:
            self.single_stats[attr] = single_stats[attr].to_dict()
        query = 'SELECT t1._vid_,t1.attribute, t1.domain, t1.object FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name is not  NULL ORDER BY _vid_;' % (
            AuxTables.cell_domain.name, AuxTables.dk_cells.name)

        results = self.engine.execute_query(query)
        for res in results:
            vid = int(res[0])
            attribute = res[1]
            domain = res[2].split('|||')
            object = res[3]
            if vid not in majority_dict:
                majority_dict[vid] = {}
                majority_dict[vid][object] = {}
                majority_dict[vid][object][attribute] = ([0], 0)
            for idx, val in enumerate(domain):
                freq = self.single_stats[attribute][(object, val)]
                if freq > majority_dict[vid][object][attribute][1]:
                    majority_dict[vid][object][attribute] = ([val], freq)
                elif freq == majority_dict[vid][object][attribute][1]:
                    majority_dict[vid][object][attribute][0].append(val)
        cell = []
        for vid in majority_dict:
            for object in majority_dict[vid]:
                for attribute in majority_dict[vid][object]:
                    if len(majority_dict[vid][object][attribute][0]) > 1:
                        value = random.choice(majority_dict[vid][object][attribute][0])
                    else:
                        value = majority_dict[vid][object][attribute][0][0]
                    app = []
                    app.append({"_vid_": vid, "object": object,
                                "attribute": attribute,
                                "rv_value": value})
                    cell.extend(app)

        df = pd.DataFrame(data=cell)
        self.generate_aux_table(AuxTables.inf_values_dom, df, store=True, index_attrs=['_vid_'])
        return

    def create_object_truth(self, method, holdout):
        """
        Creates the inferred object dataframe

        :param method: method that we use (accu, catd, slimfast)
        :param session: Holoclean session
        :param holdout: Testing data
        :return:
        """
        list_truth = []
        for object in method.object_inferred_truth:
            object_list = object.split("+_+")
            if object_list[0] in holdout:
                list_truth.append((object_list[0], object_list[1],
                                   method.object_inferred_truth[object]))
        labels = ['object', 'attribute', 'rv_value']
        df = pd.DataFrame.from_records(list_truth, columns=labels)
        self.generate_aux_table(AuxTables.inf_values_dom, df, store=True)

    def create_training(self):
        """
        This method separates the training from the testing data

        :param ratio: ratio of training with testing data
        :param session: Holoclean session
        :param schema: schema of the dataset
        """
        if self.env['test2train'] != 1:
            x_training, x_testing = train_test_split(self.aux_table[AuxTables.entity].df,
                                                     test_size=self.env['test2train'], random_state=self.seed)
            self.generate_aux_table(AuxTables.dk_cells, x_testing, store=True)
            holdout = {}

            records = x_testing.to_records()
            for row in tqdm(list(records)):
                vid = row[1]
                holdout[vid] = ""

            labelled = {}
            records = x_training.to_records()

            for row in tqdm(list(records)):
                for attribute in self.attr_to_idx:
                    vid = row[1] + "+_+" + attribute
                    labelled[vid] = self.correct_object_dict[row[1]][attribute]
        else:
            labelled = {}
            holdout = {}

            records = self.aux_table[AuxTables.entity].df.to_records()
            for row in tqdm(list(records)):
                vid = row[1]
                holdout[vid] = ""

        return labelled, holdout
