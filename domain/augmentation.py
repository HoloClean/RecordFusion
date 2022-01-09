import time
import pandas as pd
from tqdm import tqdm
from dataset import AuxTables
from collections import OrderedDict
import string
import random
from difflib import SequenceMatcher


# GM

def build_reduced_model():
    lst = []
    lst.append(("".join([string.ascii_lowercase, string.ascii_uppercase]), "A"))
    lst.append(("0123456789", "N"))
    lst.append(("-", "-"))
    lst.append(("/", "/"))
    lst.append(("(", "("))
    lst.append((")", ")"))
    lst.append(("&", "&"))
    lst.append(("$", "$"))
    lst.append(("%", "%"))
    lst.append(("_", "_"))
    lst.append(("'", "'"))
    lst.append((".", "."))
    lst.append((",", ","))
    lst.append(("#", "#"))
    lst.append(("@", "@"))
    lst.append(("!", "!"))
    lst.append((":", ":"))
    lst.append(("+", "+"))
    lst.append((";", ";"))
    lst.append(("[", "["))
    lst.append(("]", "]"))
    lst.append((" ", " "))
    lst.append(("*", "*"))
    lst.append(("?", "?"))
    lst.append(("{", "{"))
    lst.append(("}", "}"))
    lst.append(("=", "="))
    lst.append(("|", "|"))
    lst.append(("/", "/"))
    lst.append(("~", "~"))

    return OrderedDict(lst)


class Augmentation:
    """
    This class creates the augmentation for each object
    """

    def __init__(self, env, dataset):
        """
        Construct augmentation

        """
        self.env = env
        self.ds = dataset
        self.language_model = build_reduced_model()
        self.dfa_list = self.ds.dfa
        self.tid_temp = - 1

    def create_sampling(self):
        self.augmented_rows = []
        init = self.ds.get_raw_data()
        query = 'SELECT distinct object FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name is NULL;' % (
            AuxTables.cell_domain.name, AuxTables.dk_cells.name)
        res = self.ds.engine.execute_query(query)
        for tuple in tqdm(res):
            # we need this line if we want to do iterations = object_size * (balance -1)
            # object_observation = init.loc[init[self.ds.key] == tuple[0]]
            # object_size = object_observation.count()[0]
            # for i in range(0, object_size * (self.env['balance']-1)):
            self.choose_dfa(tuple[0])

        for row in tqdm(self.augmented_rows):
            for ind, attr in enumerate(self.all_attrs):
                domain = self.ds.domain[self.ds.domain['object'] == row[2]][self.ds.domain['attribute'] == attr]['domain'].str.cat()
                domain_size = \
                self.ds.domain[self.ds.domain['object'] == row[2]][self.ds.domain['attribute'] == attr]['domain_size'].tolist()[0]
                domain = domain + "|||" + row[ind + 3]
                domain_size = int(domain_size) + 1
                self.ds.domain.loc[(self.ds.domain['object'] == row[2]) & (self.ds.domain['attribute'] == attr), 'domain_size'] = domain_size
                self.ds.domain.loc[(self.ds.domain['object'] == row[2]) & (self.ds.domain['attribute'] == attr), 'domain'] = domain
        self.ds.generate_aux_table(AuxTables.cell_domain, self.ds.domain, store=True, index_attrs=['_vid_'])

        return

    def choose_dfa(self, object):
        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)
        row_augment = []
        for attr in self.all_attrs:
            initial_value = self.ds.correct_object_dict[object][attr]
            init_automaton = self.ds.dfa_list_id[attr][object][initial_value]
            end_flag = True
            query = "SELECT domain FROM %s as t1 WHERE t1.object = '%s' and attribute = '%s'" % (AuxTables.cell_domain.name, object, attr)
            res = self.ds.engine.execute_query(query)
            domain = res[0][0].split("|||")
            while end_flag:

                target_automaton = self.check_automaton(initial_value,object, attr)

                string1, self.letter_to_pos1 = self.create_string(init_automaton)
                string2, self.letter_to_pos2 = self.create_string(target_automaton)

                translation = []
                self.find_translation(string1, string2, translation)

                new_string, flag = self.generate_string(initial_value, translation, string1, string2, init_automaton)
                if new_string != initial_value:
                    end_flag = False
                    break

                if flag:

                    if len(domain) == 1:
                        end_flag = False
                        break
                    domain.remove(initial_value)
                    initial_value = str(random.sample(domain, 1)[0])

                    init_automaton = self.ds.dfa_list_id[attr][object][initial_value]

            row_augment.append(new_string)
        new_row = [self.tid_temp, 'emtpy', object]
        self.tid_temp = self.tid_temp - 1
        for element in row_augment:
            new_row.append(element)
        self.augmented_rows.append(new_row)
        return

    def state_substring_match(self, init_value, first_machine, init_automation):
        zip = []
        current_position = 0
        previous_state = None
        cluster = []
        string_key_final = None
        i = 0
        cluster_index = []
        init_state = 0
        for index, char in enumerate(init_value):
            key = self.char_to_key(char)
            current_position_temp = init_automation[current_position][key]
            string_key = self.letter_to_pos1[current_position_temp]
            if previous_state != current_position_temp:
                if previous_state is None:
                    previous_state = current_position_temp
                    string_key_final = string_key
                    cluster.append(char)
                else:
                    zip.append([string_key_final, "".join(cluster), (init_state, index - 1)])
                    cluster = []
                    cluster_index = []

                    previous_state = current_position_temp
                    string_key_final = string_key
                    cluster.append(char)
                    init_state = index
            else:
                cluster.append(char)
            current_position = current_position_temp
        zip.append([string_key_final, "".join(cluster), (init_state, len(init_value) - 1)])
        return zip

    def do_translation(self, zip_state, translation):
        flag = False
        actual_index = {}
        for r in range(len(zip_state)):
            actual_index[r] = zip_state[r] + [r]
        multi_state_actual_index = self.state_trans_2_str(actual_index, translation)
        for tr in translation:
            len_of_multi_state = len(tr[0])
            if 'A' not in tr[1] and 'B' not in tr[1] and 'N' not in tr[1] and 'O' not in tr[1]:
                start_index = tr[2][0]
                for row_index in multi_state_actual_index:
                    if start_index in row_index[3]:
                        row_index[1] = tr[1]
        final_string = ''
        for index in range(0, len(zip_state)):
            for row in multi_state_actual_index:
                if index in row[3]:
                    final_string += row[1]
                    row[3] = [-1]

        if len(zip_state) == 1:
            flag = True

        return final_string, flag

    def state_trans_2_str(self, actual_index, translation):
        multi_state_actual_index = []
        used_key = []
        for tr in translation:
            len_of_multi_state = len(tr[0])
            start_index = tr[2][0]
            temp_multi_state = ''
            temp_multi_string = ''
            temp_multi_range = [0, 0]
            temp_multi_indices = []
            first_flag = True
            for key in range(start_index, start_index + len_of_multi_state):
                used_key.append(key)
                temp_multi_state += actual_index[key][0]
                temp_multi_string += actual_index[key][1]
                temp_multi_indices.append(actual_index[key][3])
                if first_flag:
                    temp_multi_range[0] = actual_index[key][2][0]
                    first_flag = False
                temp_multi_range[1] = actual_index[key][2][1]
            multi_state_actual_index.append([temp_multi_state, temp_multi_string, temp_multi_range, temp_multi_indices])
        for idx, el in enumerate(actual_index):
            if idx not in used_key:
                multi_state_actual_index.append(actual_index[idx][:3] + [[actual_index[idx][3]]])
        return multi_state_actual_index

    def generate_string(self, init_value, translation, first_machine, target_machine, init_automaton):
        state_string_zip = self.state_substring_match(init_value, first_machine, init_automaton)
        final_string, flag = self.do_translation(state_string_zip, translation)

        return final_string, flag

    def find_translation(self, initial_automaton, final_automaton, translation, position=(0, 0)):
        parts = self.lcs_split(initial_automaton, final_automaton, position)
        if len(parts[2][0]) + len(parts[1][0]) + len(parts[3]) + len(parts[4]) <= 0:
            return
        if len(parts[0]) <= 0:
            translation.append([initial_automaton, final_automaton, position])
        else:
            best_matches = self.cofusion_similarity(parts)
            for match in best_matches:
                self.find_translation(match[0][0], match[1], translation, match[0][1])

    def lcs_split(self, string1, string2, parent_range=(0, 0)):
        match = SequenceMatcher(a=string1, b=string2).find_longest_match(0, len(string1), 0, len(string2))
        lcs = string1[match.a: match.a + match.size]

        if len(lcs) > 0:
            first_first = [[string1.split(lcs, 1)[0], (0 + parent_range[0], match.a + parent_range[0])]]
            first_second = [[string1.split(lcs, 1)[1], (match.a + match.size + parent_range[0],
                                                        len(string1) - 1 + parent_range[0])]]
            return [lcs] + first_first + first_second + string2.split(lcs, 1)
        else:
            return [lcs] + [['', (0, 0)]] + [[string1, (0, len(string1) - 1)]] + ['', string2]

    def cofusion_similarity(self, parts):
        sub_strings = parts[1:]
        if self.str_similarity(sub_strings[0][0], sub_strings[2]) + self.str_similarity(sub_strings[1][0], sub_strings[
            3]) >= self.str_similarity(sub_strings[0][0], sub_strings[3]) + self.str_similarity(sub_strings[1][0],
                                                                                                sub_strings[2]):
            return [[sub_strings[0], sub_strings[2]], [sub_strings[1], sub_strings[3]]]
        else:
            return [[sub_strings[0], sub_strings[3]], [sub_strings[1], sub_strings[2]]]

    def str_similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def create_string(self, automaton):
        string_list = []
        letter_to_position = {}
        previous_char = None
        for i in range(0, len(automaton)):
            flag2 = False
            for key in automaton[i].keys():
                if key is not None:
                    if key == previous_char:
                        if key == 'A' or key == 'N':
                            string_list[-1] = chr(ord(string_list[-1]) + 1)
                            previous_char = string_list[-1]
                    else:
                        previous_char_temp = key
                        flag2 = True
            if flag2 == True:
                string_list.append(previous_char_temp)
                previous_char = previous_char_temp
        for i in range(0, len(string_list)):
            if len(string_list) == len(automaton):
                letter_to_position[i + 1] = string_list[i]
            else:
                letter_to_position[0] = string_list[0]
                letter_to_position[i + 1] = string_list[i]

        return "".join(string_list), letter_to_position

    def check_automaton(self, initial, object, attr):

        query = "SELECT distinct object FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name  is NULL and t1.object != '%s';" % (
            AuxTables.cell_domain.name, AuxTables.dk_cells.name, object)
        res1 = self.ds.engine.execute_query(query)


        flag_choose = True
        while flag_choose:
            except_object = str(random.sample(res1, 1)[0][0])
            query = "SELECT domain FROM %s as t1 WHERE t1.object = '%s' and attribute = '%s'" % (AuxTables.cell_domain.name, except_object, attr)
            res = self.ds.engine.execute_query(query)
            domain = res[0][0].split("|||")
            val = str(random.sample(domain, 1)[0])
            automaton = self.ds.dfa_list_id[attr][except_object][val]
            if not self.check(initial, automaton):
                flag_choose = False
        self.second_val = val
        return automaton

    def check(self, correct_value, automaton):
        value = self.get_key(correct_value)
        flag = True
        current_state = 0
        for char in value:
            try:
                current_state = automaton[current_state][char]
            except:
                flag = False
        if current_state < (len(automaton) - 1):
            flag = False
        return flag

    def get_key(self, val):
        """
        for each val we get the reduce langauge model
        :param val:
        :return:
        """
        subkeys = [self.char_to_key(char) for char in val]
        return "".join(subkeys)

    def char_to_key(self, char):
        """
        for each char we get the reduce language model val
        """
        for dict_key in self.language_model.keys():
            if char in dict_key:
                return self.language_model[dict_key]
