import time
import pandas as pd
from tqdm import tqdm
from dataset import AuxTables

#GM
class Objects:
    """
    This class creates the object and the random variables from the Init dataset
    """

    def __init__(self, env, dataset):
        """
        Construct Objects object

        """
        self.env = env
        self.ds = dataset

    def setup(self):
        tic = time.time()
        try:
            self.ds.domain = self._create_object_table()
            self.store_domains(self.ds.domain)
            status = "DONE with domain preparation and creation of objects."
        except Exception as e:
             status = "ERROR setting up the objectss: %s"%str(e)
        toc = time.time()
        return status, toc - tic

    def _create_object_table(self):
        """
        This method creates the tables for all the objects
        """
        object_dict = {}
        object_to_tid = {}

        records = self.ds.get_raw_data().to_records()
        self.all_attrs = list(records.dtype.names)
        for row in tqdm(list(records)):
            object_key = row[self.ds.key]
            if object_key not in object_dict:
                object_dict[object_key] = {}
                object_to_tid[object_key] = len(object_to_tid)

            for attr in self.all_attrs:
                if attr!= self.ds.key and attr != self.ds.src and attr != 'index' and attr !='_tid_':
                    if attr not in object_dict[object_key]:
                        object_dict[object_key][attr] = {}
                    if row[attr] not in object_dict[object_key][attr]:
                        object_dict[object_key][attr][row[attr]] = 0
                    else:
                        object_dict[object_key][attr][row[attr]] =  \
                            object_dict[object_key][attr][row[attr]] +1
        vid = 0
        cells = []
        entity =[]
        for object in object_dict:
            app = []
            entity.append(object)
            for attr in object_dict[object]:
                tid = object_to_tid[object]
                correct_value = self.get_correct_value(attr, object)
                cid = self.ds.get_cell_id(tid, attr)
                dom = []
                for attr_val in object_dict[object][attr]:
                    if attr_val != "_nan_" and len(attr_val)>0 :
                        dom.append(attr_val)
                correct_value_idx = dom.index(correct_value)

                app.append({"_tid_": tid, "attribute": attr, "_cid_": cid,
                            "_vid_": vid, "domain": "|||".join(dom),
                            "domain_size": len(dom),"correct_value": correct_value,"correct_value_idx": correct_value_idx, "object":object})
                vid += 1

            cells.extend(app)

        domain_df = pd.DataFrame(data=cells)
        entity_df = pd.DataFrame(entity, columns=['entity_name'])

        self.ds.generate_aux_table(AuxTables.entity, entity_df, store=True)

        if self.env['statistics']:
            self.print_statistics()

        return domain_df


    def get_correct_value(self, attr, object):
        return self.ds.correct_object_dict[object][attr]


    def store_domains(self, domain):
        if domain.empty:
            raise Exception("ERROR: Generated domain is empty.")
        else:
            self.ds.generate_aux_table(AuxTables.cell_domain, domain, store=True )
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_tid_'])
            self.ds.aux_table[AuxTables.cell_domain].create_db_index(self.ds.engine, ['_cid_'])
            query = "SELECT object,_vid_, _cid_, _tid_, attribute, a.rv_val, a.val_id from %s , unnest(string_to_array(regexp_replace(domain,\'[{\"\"}]\',\'\',\'gi\'),\'|||\')) WITH ORDINALITY a(rv_val,val_id)" % AuxTables.cell_domain.name
            self.ds.generate_aux_table_sql(AuxTables.pos_values, query, index_attrs=['object', '_tid_', 'attribute'])

    def print_statistics(self):
        query = "SELECT count(*) FROM %s " %(self.ds.raw_data.name)
        res = self.ds.engine.execute_query(query)
        total_observations = float(res[0][0])
        print('total observ:'+str(total_observations))
        query = "SELECT count(distinct src) FROM %s " %(self.ds.raw_data.name)
        res = self.ds.engine.execute_query(query)
        src_number = float(res[0][0])
        print('src:'+str(src_number))
        query = "SELECT count(distinct %s) FROM %s " %(self.ds.key, self.ds.raw_data.name)
        res = self.ds.engine.execute_query(query)
        object_number = float(res[0][0])
        print('objects:'+str(object_number))

        query = "SELECT * FROM %s " %(self.ds.raw_data.name)
        results = self.ds.engine.execute_query(query)

        feature = {}
        src = {}
        object = {}
        for res in results:
            if res[2] not in object:
                object[res[2]]= 1
            else:
                object[res[2]] = object[res[2]] + 1
            if res['src'] not in src:
                src[res['src']] =1
            else:
                src[res['src']] =  src[res['src']] + 1
            for i in range(3,len(res)):
                if i not in feature:
                    feature[i]={}
                else:
                    if res[i] not in feature[i]:
                        feature[i][res[i]] ={}
        total = 0
        for obj in object:
            total = total + object[obj]
        total = float(total) / len(object)
        print('Avgobjects:'+str(total))

        total = 0
        for source in src:
            total = total + src[source]
        total = float(total) / len(src)
        print('Avgsrc:'+str(total))

        avg_feature = 0
        for feat in feature:
            avg_feature = avg_feature + len(feature[feat])
        avg_feature = float(avg_feature) / len(feature)
        print('Avgfeature'+str(avg_feature))
        print('Features:'+str(len(feature)))
        return
