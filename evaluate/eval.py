import logging
import os
from string import Template
import time

import pandas as pd

from dataset import AuxTables
from dataset.table import Table, Source

errors_template = Template('SELECT count(*) ' \
                           'FROM  "$init_table" as t1, "$grdt_table" as t2 ' \
                           'WHERE t1._tid_ = t2._tid_ ' \
                           '  AND t2._attribute_ = \'$attr\' ' \
                           '  AND t1."$attr" != t2._value_')

"""
The 'errors' aliased subquery returns the (_tid_, _attribute_, _value_)
from the ground truth table for all cells that have an error in the original
raw data.

The 'repairs' aliased table contains the cells and values we've inferred.

We then count the number of cells that we repaired to the correct ground
truth value.
"""
correct_repairs_template = Template('SELECT COUNT(*) FROM '
                                    '  (SELECT t2._tid_, t2._attribute_, t2._value_ '
                                    '     FROM "$init_table" as t1, "$grdt_table" as t2 '
                                    '    WHERE t1._tid_ = t2._tid_ '
                                    '      AND t2._attribute_ = \'$attr\' '
                                    '      AND t1."$attr" != t2._value_ ) as errors, $inf_dom as repairs '
                                    'WHERE errors._tid_ = repairs._tid_ '
                                    '  AND errors._attribute_ = repairs.attribute '
                                    '  AND errors._value_ = repairs.rv_value')


class EvalEngine:
    def __init__(self, env, dataset):
        self.env = env
        self.ds = dataset

    def load_data(self, name, fpath, tid_col, attr_col, val_col, na_values=None):
        tic = time.clock()
        try:
            raw_data = pd.read_csv(fpath, na_values=na_values, encoding='utf-8')
            raw_data.fillna('_nan_', inplace=True)
            raw_data.rename({tid_col: '_tid_',
                             attr_col: '_attribute_',
                             val_col: '_value_'},
                            axis='columns',
                            inplace=True)
            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]
            # Normalize string to whitespaces.
            raw_data['_value_'] = raw_data['_value_'].str.strip().str.lower()
            self.clean_data = Table(name, Source.DF, df=raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('load_data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def evaluate_repairs(self):
        self.compute_total_repairs()
        self.compute_total_repairs_grdt()
        self.compute_total_errors()
        self.compute_detected_errors()
        self.compute_correct_repairs()
        prec = self.compute_precision()
        rec = self.compute_recall()
        rep_recall = self.compute_repairing_recall()
        f1 = self.compute_f1()
        rep_f1 = self.compute_repairing_f1()

        if self.env['verbose']:
            self.log_weak_label_stats()

        return prec, rec, rep_recall, f1, rep_f1

    def eval_report(self):
        tic = time.clock()
        try:
            prec, rec, rep_recall, f1, rep_f1 = self.evaluate_repairs()
            report = "Precision = %.2f, Recall = %.2f, Repairing Recall = %.2f, F1 = %.2f, Repairing F1 = %.2f, Detected Errors = %d, Total Errors = %d, Correct Repairs = %d, Total Repairs = %d, Total Repairs on correct cells (Grdth present) = %d, Total Repairs on incorrect cells (Grdth present) = %d" % (
                      prec, rec, rep_recall, f1, rep_f1,
                      self.detected_errors, self.total_errors, self.correct_repairs,
                      self.total_repairs, self.total_repairs_grdt_correct, self.total_repairs_grdt_incorrect)
            report_list = [prec, rec, rep_recall, f1, rep_f1, self.detected_errors, self.total_errors,
                           self.correct_repairs, self.total_repairs, self.total_repairs_grdt]
        except Exception as e:
            logging.error("ERROR generating evaluation report %s" % e)
            raise

        toc = time.clock()
        report_time = toc - tic
        return report, report_time, report_list

    def compute_total_repairs(self):
        """
        compute_total_repairs memoizes the number of repairs:
        the # of cells that were inferred and where the inferred value
        is not equal to the initial value.
        """

        query = "SELECT count(*) FROM " \
                "  (SELECT _vid_ " \
                "     FROM {} as t1, {} as t2 " \
                "    WHERE t1._tid_ = t2._tid_ " \
                "      AND t1.attribute = t2.attribute " \
                "      AND t1.init_value != t2.rv_value) AS t".format(AuxTables.cell_domain.name,
                                                                      AuxTables.inf_values_dom.name)
        res = self.ds.engine.execute_query(query)
        self.total_repairs = float(res[0][0])

    def compute_total_repairs_grdt(self):
        """
        compute_total_repairs_grdt memoizes the number of repairs for cells
        that are specified in the clean/ground truth data. Otherwise repairs
        are defined the same as compute_total_repairs.

        We also distinguish between repairs on correct cells and repairs on
        incorrect cells (correct cells are cells where init == ground truth).
        """
        query = """
        SELECT
            (t1.init_value = t3._value_) AS is_correct,
            count(*)
        FROM   {} as t1, {} as t2, {} as t3
        WHERE  t1._tid_ = t2._tid_
          AND  t1.attribute = t2.attribute
          AND  t1.init_value != t2.rv_value
          AND  t1._tid_ = t3._tid_
          AND  t1.attribute = t3._attribute_
        GROUP BY is_correct
          """.format(AuxTables.cell_domain.name,
                  AuxTables.inf_values_dom.name,
                  self.clean_data.name)
        res = self.ds.engine.execute_query(query)

        # Memoize the number of repairs on correct cells and incorrect cells.
        # Since we do a GROUP BY we need to check which row of the result
        # corresponds to the correct/incorrect counts.
        self.total_repairs_grdt_correct, self.total_repairs_grdt_incorrect = 0, 0
        self.total_repairs_grdt = 0
        if not res:
            return

        if res[0][0]:
            correct_idx, incorrect_idx = 0, 1
        else:
            correct_idx, incorrect_idx = 1, 0
        if correct_idx < len(res):
            self.total_repairs_grdt_correct = float(res[correct_idx][1])
        if incorrect_idx < len(res):
            self.total_repairs_grdt_incorrect =  float(res[incorrect_idx][1])
        self.total_repairs_grdt = self.total_repairs_grdt_correct + self.total_repairs_grdt_incorrect

    def compute_total_errors(self):
        """
        compute_total_errors memoizes the number of cells that have a
        wrong initial value: requires ground truth data.
        """
        queries = []
        total_errors = 0.0
        for attr in self.ds.get_attributes():
            query = errors_template.substitute(init_table=self.ds.raw_data.name,
                                               grdt_table=self.clean_data.name,
                                               attr=attr)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            total_errors += float(res[0][0])
        self.total_errors = total_errors

    def compute_detected_errors(self):
        """
        compute_detected_errors memoizes the number of error cells that
        were detected in error detection: requires ground truth.

        This value is always equal or less than total errors (see
        compute_total_errors).
        """
        query = "SELECT count(*) FROM " \
                "  (SELECT _vid_ " \
                "   FROM   %s as t1, %s as t2, %s as t3 " \
                "   WHERE  t1._tid_ = t2._tid_ AND t1._cid_ = t3._cid_ " \
                "     AND  t1.attribute = t2._attribute_ " \
                "     AND  t1.init_value != t2._value_) AS t" \
                % (AuxTables.cell_domain.name, self.clean_data.name, AuxTables.dk_cells.name)
        res = self.ds.engine.execute_query(query)
        self.detected_errors = float(res[0][0])

    def compute_correct_repairs(self):
        """
        compute_correct_repairs memoizes the number of error cells
        that were correctly inferred.

        This value is always equal or less than total errors (see
        compute_total_errors).
        """
        queries = []
        correct_repairs = 0.0
        for attr in self.ds.get_attributes():
            query = correct_repairs_template.substitute(init_table=self.ds.raw_data.name, grdt_table=self.clean_data.name,
                                                        attr=attr, inf_dom=AuxTables.inf_values_dom.name)
            queries.append(query)
        results = self.ds.engine.execute_queries(queries)
        for res in results:
            correct_repairs += float(res[0][0])
        self.correct_repairs = correct_repairs

    def compute_recall(self):
        """
        Computes the recall (# of correct repairs / # of total errors).
        """
        if self.total_errors == 0:
            return 0
        return self.correct_repairs / self.total_errors

    def compute_repairing_recall(self):
        """
        Computes the _repairing_ recall (# of correct repairs / # of total
        _detected_ errors).
        """
        if self.detected_errors == 0:
            return 0
        return self.correct_repairs / self.detected_errors

    def compute_precision(self):
        """
        Computes precision (# correct repairs / # of total repairs w/ ground truth)
        """
        if self.total_repairs_grdt == 0:
            return 0
        return self.correct_repairs / self.total_repairs_grdt

    def compute_f1(self):
        prec = self.compute_precision()
        rec = self.compute_recall()
        if prec+rec == 0:
            return 0
        f1 = 2*(prec*rec)/(prec+rec)
        return f1

    def compute_repairing_f1(self):
        prec = self.compute_precision()
        rec = self.compute_repairing_recall()
        if prec+rec == 0:
            return 0
        f1 = 2*(prec*rec)/(prec+rec)
        return f1

    def log_weak_label_stats(self):
        query = """
        select
            (t3._tid_ is NULL) as clean,
            (t1.fixed) as status,
            (t4._tid_ is NOT NULL) as inferred,
            (t1.init_value = t2._value_) as init_eq_grdth,
            (t1.init_value = t4.rv_value) as init_eq_infer,
            (t1.weak_label = t1.init_value) as wl_eq_init,
            (t1.weak_label = t2._value_) as wl_eq_grdth,
            (t1.weak_label = t4.rv_value) as wl_eq_infer,
            (t2._value_ = t4.rv_value) as infer_eq_grdth,
            count(*) as count
        from
            {cell_domain} as t1,
            {clean_data} as t2
            left join {dk_cells} as t3 on t2._tid_ = t3._tid_ and t2._attribute_ = t3.attribute
            left join {inf_values_dom} as t4 on t2._tid_ = t4._tid_ and t2._attribute_ = t4.attribute where t1._tid_ = t2._tid_ and t1.attribute = t2._attribute_
        group by
            clean,
            status,
            inferred,
            init_eq_grdth,
            init_eq_infer,
            wl_eq_init,
            wl_eq_grdth,
            wl_eq_infer,
            infer_eq_grdth
        """.format(cell_domain=AuxTables.cell_domain.name,
                clean_data=self.clean_data.name,
                dk_cells=AuxTables.dk_cells.name,
                inf_values_dom=AuxTables.inf_values_dom.name)

        res = self.ds.engine.execute_query(query)

        df_stats = pd.DataFrame(res,
                columns=["is_clean", "cell_status", "is_inferred",
                    "init = grdth", "init = inferred",
                    "w. label = init", "w. label = grdth", "w. label = inferred",
                    "infer = grdth", "count"])
        df_stats = df_stats.sort_values(list(df_stats.columns)).reset_index(drop=True)
        logging.debug("weak label statistics:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', len(df_stats))
        pd.set_option('display.max_colwidth', -1)
        logging.debug("%s", df_stats)
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')

    #------fusion
    # GM

    def load_data(self, name, fpath, tid_col, attr_col, val_col, na_values=None):
        tic = time.clock()
        try:
            raw_data = pd.read_csv(fpath, na_values=na_values, encoding='utf-8')
            raw_data.fillna('_nan_', inplace=True)
            raw_data.rename({tid_col: '_tid_',
                             attr_col: '_attribute_',
                             val_col: '_value_'},
                            axis='columns',
                            inplace=True)
            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]
            # Normalize string to whitespaces.
            raw_data['_value_'] = raw_data['_value_'].str.strip().str.lower()
            self.clean_data = Table(name, Source.DF, df=raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('load_data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time

    def load_data_fusion(self, name, fpath, tid_col, attr_col, val_col, na_values=None):
        tic = time.clock()
        try:
            #everything is a string
            raw_data = pd.read_csv(fpath, dtype=object, na_values=na_values, encoding='utf-8')
            raw_data.fillna('_nan_', inplace=True)
            raw_data.rename({tid_col: '_tid_',
                             attr_col: '_attribute_',
                             val_col: '_value_'},
                            axis='columns',
                            inplace=True)
            raw_data = raw_data[['_tid_', '_attribute_', '_value_']]
            # Normalize string to whitespaces.

            raw_data['_value_'] = raw_data['_value_'].str.strip().str.lower()

            raw_data['_tid_'] = raw_data['_tid_'].str.strip().str.lower()

            self.clean_data = Table(name, Source.DF, df=raw_data)
            self.clean_data.store_to_db(self.ds.engine.engine)
            self.clean_data.create_db_index(self.ds.engine, ['_tid_'])
            self.clean_data.create_db_index(self.ds.engine, ['_attribute_'])
            object_dict = {}
            records = self.clean_data.df.to_records()
            self.all_attrs = list(records.dtype.names)
            for row in list(records):
                object_key = str(row['_tid_'])
                if object_key not in object_dict:
                    object_dict[object_key] = {}
                object_dict[object_key][row['_attribute_']]= str(row['_value_'])
            self.ds.correct_object_dict = object_dict
            status = 'DONE Loading {fname}'.format(fname=os.path.basename(fpath))
        except Exception:
            logging.error('load_data for table %s', name)
            raise
        toc = time.clock()
        load_time = toc - tic
        return status, load_time




    def eval_report_fusion(self):
        tic = time.clock()
        try:
            self.ds.prec = self.evaluate_repairs_fusion()
            report = "Precision of iteration is  = %.4f" % (self.ds.prec)

        except Exception as e:
            report = "ERROR generating evaluation report: %s" % str(e)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time


    def eval_report_fusion_recurr(self, validation):
        tic = time.clock()
        try:
            f =open('prec.txt','a+')
            self.ds.prec = self.evaluate_repairs_fusion_recurr()
            if validation:
                report = "Precision of validation set = %.4f" % (self.ds.prec)

            else:
                report = "Precision = %.4f" % (self.ds.prec)
            f.write('Precision' + str(self.ds.prec))
            f.close()
        except Exception as e:
            report = "ERROR generating evaluation report: %s" % str(e)
        toc = time.clock()
        report_time = toc - tic
        return report, report_time

    # GM
    def evaluate_repairs_fusion(self):

        # we get the accuracy for each attribute
        for attribute in self.ds.raw_data.df.columns:
            if attribute!="_tid_" and attribute!=self.ds.src and attribute != self.ds.key:
                query = "select count(*) from  %s AS t1 where attribute= '%s'" % (
                    AuxTables.inf_values_dom.name, attribute)
                total_repair = self.ds.engine.execute_query(query)

                # count number of instances in which inferred values match the ground truth
                query = "select count(*) from  %s AS t1, %s as t2 where t1.object = cast(t2._tid_ as varchar(100)) and t1.attribute = t2._attribute_ and t1.rv_value = t2._value_ and t1.attribute = '%s'" % (
                    AuxTables.inf_values_dom.name, self.clean_data.name, attribute)
                correct = self.ds.engine.execute_query(query)
                # return precision based on count
                prec = float(correct[0][0]) / float(total_repair[0][0])
                print("The precision of the attribute %s is: %s" % (attribute, str(prec)))


        #we get the accuracy for all the predictions

        query = "select count(*) from  %s AS t1" % (AuxTables.inf_values_dom.name)
        total_repair = self.ds.engine.execute_query(query)

        # count number of instances in which inferred values match the ground truth
        query = "select count(*) from  %s AS t1, %s as t2 where t1.object = cast(t2._tid_ as varchar(100)) and t1.attribute = t2._attribute_ and t1.rv_value = t2._value_" % (
        AuxTables.inf_values_dom.name, self.clean_data.name)
        correct = self.ds.engine.execute_query(query)

        # return precision based on count
        prec = float(correct[0][0]) / float(total_repair[0][0])
        return prec


    def evaluate_repairs_fusion_recurr(self):


        #we get the accuracy for all the predictions

        query = "select count(*) from  %s AS t1" % (AuxTables.inf_values_dom.name)
        total_repair = self.ds.engine.execute_query(query)

        # count number of instances in which inferred values match the ground truth
        query = "select count(*) from  %s AS t1, %s as t2 where t1.object = cast(t2._tid_ as varchar(100)) and t1.attribute = t2._attribute_ and t1.rv_value = t2._value_" % (
        AuxTables.inf_values_dom.name, self.clean_data.name)
        correct = self.ds.engine.execute_query(query)

        # return precision based on count
        prec = float(correct[0][0]) / float(total_repair[0][0])
        return prec


