import time
import pandas as pd
from tqdm import tqdm
from dataset import AuxTables
from collections import OrderedDict
import string
from copy import  copy


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


class DFA_creation:
    """
    This class creates the DFA object and the random variables from the Init dataset
    """

    def __init__(self, env, dataset):
        """
        Construct DFA object

        """
        self.env = env
        self.ds = dataset
        self.language_model = build_reduced_model()

    def setup(self):
        tic = time.time()
        self.ds.dfa = self._create_dfa()
        status = "DONE with creation  of  dfa objects."
        toc = time.time()
        return status, toc - tic

    def _create_dfa(self):
        """
        This method creates the dfa for all the objects
        """
        query = 'SELECT object, attribute , domain FROM %s AS t1 LEFT JOIN %s AS t2 ON t1.object = t2.entity_name WHERE t2.entity_name is NULL;' % (
        AuxTables.cell_domain.name, AuxTables.dk_cells.name)

        res = self.ds.engine.execute_query(query)

        self.all_attrs = self.ds.get_attributes()
        self.all_attrs.remove(self.ds.src)
        self.all_attrs.remove(self.ds.key)

        dfa_list = {}
        self.ds.dfa_list_id = {}
        self.ds.key_to_dfa = {}
        for tuple in tqdm(res):
            object = tuple[0]
            attr = tuple[1]
            domain = tuple[2].split('|||')
            if attr not in dfa_list:
                dfa_list[attr] = {}
                self.ds.dfa_list_id[attr] = {}
                self.ds.key_to_dfa[attr] ={}
            if object not in dfa_list[attr]:
                dfa_list[attr][object] = []
                self.ds.dfa_list_id[attr][object]={}
            for val in domain:
                key = self.get_key(val)
                if  val == 'h5':
                    pass
                auto = self.create_automaton(key, dfa_list[attr][object])

                if auto is not None:
                    self.ds.dfa_list_id[attr][object][val] = copy(auto)
                    dfa_list[attr][object].append(auto)
                else:
                    self.ds.dfa_list_id[attr][object][val] = copy(self.temp)
        return dfa_list

    def create_automaton(self, key, dfa_list):
        """
        for each key value we get a new automaton or none if a automaton exist
        :param key: key value
        :param dfa_list: a list with all the automaton that we have created
        :return: automaton or NOne if the automaton already exist
        """
        if len(dfa_list) == 0:
            new_automaton = self.create_new(key)
        else:
            new_automaton = self.create_from_list(key, dfa_list)
        return new_automaton

    def create_new(self, key):
        """
        create a new automaton from the begining
        :param key: key value that we will use it to create the automaton
        :return: automaon
        """
        automaton = {}
        prev_char = -1
        position = 0
        automaton[position] = {}

        for char in key:
            if prev_char != char:
                automaton[position][char] = position + 1
                prev_char = char
                position = position + 1
                automaton[position] = {}
            else:
                automaton[position][char] = position
        return automaton

    def create_from_list(self, key, dfa_list):
        """
        check if the key value fits to a automaton from the list or update the list with a new automaton from the position
        that it does not fit
        :param key: key value
        :param dfa_list: list with all the automaton for the object/attribute
        :return: automaton
        """
        for dfa_index in range(0, len(dfa_list)):
            maximum = 0
            maximum_index = 0
            lenght = 0
            position = 0
            old_position = None
            fit = True
            for char in key:
                try:
                    new_position = dfa_list[dfa_index][position][char]
                    old_position = position
                    position = new_position
                    lenght = lenght + 1
                except:
                    if lenght > maximum:
                        maximum_index = dfa_index
                        maximum = lenght
                    fit = False
                    break
            if fit:
                # if it goes to the last position
                final_lenght= 0
                if len(dfa_list[dfa_index][len(dfa_list[dfa_index])-1])==0:
                    final_lenght = len(dfa_list[dfa_index]) -1
                else:
                    final_lenght = len(dfa_list[dfa_index])
                if position == final_lenght:
                    self.temp = dfa_list[dfa_index]
                    return None
                else:
                    automaton= self.create_new(key)
                    # if fits but does not go to the last position

                    # automaton = {}
                    # if position == old_position:
                    #     final = position + 1
                    # else:
                    #     final = position
                    #
                    # for index in range(0, final):
                    #     automaton[index] = copy(dfa_list[dfa_index][index])
                    # deleted_keys = []
                    # for key in automaton[len(automaton)-1]:
                    #     if automaton[len(automaton)-1][key] > (len(automaton)-1):
                    #         if len(automaton[len(automaton)-1][key]) == 1:
                    #             automaton[len(automaton) - 1][key] = {}
                    #         else:
                    #             deleted_keys.append(key)
                    # for key in deleted_keys:
                    #     del automaton[len(automaton)-1][key]
                    # if len(automaton[len(automaton)-1]) == 0:
                    #     del  automaton[len(automaton)-1]

                    return automaton

        if maximum == 0:
            # if we do not go to the first position we create a new automaton from the beginning
            automaton = self.create_new(key)
        else:
            # we update an automaton from where we get an error
            automaton = self.update_automaton(key, dfa_list, maximum_index)
        return automaton

    def update_automaton(self, key, dfa_list, maximum_index):
        """
        we create a new automaton by updating another automaton from the position that we get an error
        :param key: key value
        :param dfa_list:  list with the automaton
        :param maximum_index:  index of the automaton that we will use
        :return: automaton
        """
        position = 0
        automaton = {}
        automaton[position] = {}
        fit = True
        prev_char = - 1

        for char in key:
            if fit:
                try:
                    # the new automaton will be the same as the old automaton until we get an error
                    if prev_char != char:
                        automaton[position][char] = \
                        dfa_list[maximum_index][position][char]
                        prev_char = char
                        position = position + 1
                        automaton[position] = {}
                    else:
                        automaton[position][char] = position
                except:
                    # we continue by creating new position for the new automaton

                    fit = False
                    if prev_char != char:
                        automaton[position][char] = position + 1
                        prev_char = char
                        position = position + 1
                        automaton[position] = {}
                    else:
                        automaton[position][char] = position
            else:
                # we continue by creating new position for the new automaton
                if prev_char != char:
                    automaton[position][char] = position + 1
                    prev_char = char
                    position = position + 1
                    automaton[position] = {}
                else:
                    automaton[position][char] = position
        return automaton

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
