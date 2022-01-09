import random
import math
from scipy.stats import chi2
import operator
from tqdm import tqdm


class CATD:
	def __init__(self, object_truth, source_observations,alpha):
                # load data
                self.object_truth = object_truth
                self.source_observations = source_observations
		self.alpha = alpha

                # init local dictionaries
                self.object_inferred_truth = {}
                self.object_distinct_observations = {}
                self.object_observations = {}
                # object to value dictionary
                for source_id in self.source_observations:
                        for oid in self.source_observations[source_id]:
				if oid in object_truth:
                                        continue
                                if oid not in self.object_observations:
                                        self.object_observations[oid] = []
                                        self.object_distinct_observations[oid] = set([])
                                        self.object_inferred_truth[oid] = source_observations[source_id][oid]
                                self.object_observations[oid].append((source_id, source_observations[source_id][oid]))
                                self.object_distinct_observations[oid].add(source_observations[source_id][oid])

                # initialize source accuracy --- utilize any ground truth if specified
                self.source_accuracy = {}
                self._init_src_weights()

	def _normalize_weights(self):
		total_weight = 0.0
		for src in self.source_accuracy:
			total_weight += self.source_accuracy[src]
		for src in self.source_accuracy:
			self.source_accuracy[src] /= total_weight

        def _init_src_weights(self):
                for source_id in self.source_observations:
                        errors = 0.0
                        total = 0.0
                        for oid in self.source_observations[source_id]:
                                if oid in self.object_truth:
                                        total += 1.0
					self.object_inferred_truth[oid] = self.object_truth[oid]
                                        if self.object_truth[oid] != self.source_observations[source_id][oid]:
                                                errors += 1.0
                        if total == 0.0:
                                self.source_accuracy[source_id] =  round(random.uniform(0,1),3)
                        else:
				if errors == 0.0:
                                        errors = 0.001
                                self.source_accuracy[source_id] = chi2.ppf(self.alpha,total)/errors
		self._normalize_weights()

        def update_object_assignment(self):
                for oid in self.object_observations:
                        obs_scores = {}
                        for (src_id, value) in self.object_observations[oid]:
                                if value not in obs_scores:
                                        obs_scores[value] = 0.0
                                obs_scores[value] += self.source_accuracy[src_id]

                        # assign largest score
                        self.object_inferred_truth[oid] = max(obs_scores, key=obs_scores.get)

        def update_source_weights(self):
                for source_id in self.source_observations:
                        errors = 0.0
                        total  = 0.0
                        for oid in self.source_observations[source_id]:
                                if oid in self.object_inferred_truth:
                                        total += 1.0
                                        if self.object_inferred_truth[oid] != self.source_observations[source_id][oid]:
                                                errors += 1.0
                        assert total != 0.0
			if errors == 0.0:
				errors = 0.001
                        self.source_accuracy[source_id] = chi2.ppf(self.alpha,total)/errors
		self._normalize_weights()


        def solve(self, iterations=100):
                for i in tqdm(range(iterations)):
                        self.update_object_assignment()
                        self.update_source_weights()


class Accu:
    """
    Accu algorithm
    """
    def __init__(self, object_truth, source_observations):
        # load data
        self.object_truth = object_truth
        self.source_observations = source_observations

        # init local dictionaries
        self.object_inferred_truth = {}
        self.object_distinct_observations = {}
        self.object_observations = {}
        # object to value dictionary
        for source_id in self.source_observations:
            for oid in self.source_observations[source_id]:
                if oid in object_truth:
                    continue
                if oid not in self.object_observations:
                    self.object_observations[oid] = []
                    self.object_distinct_observations[oid] = set([])
                    self.object_inferred_truth[oid] = source_observations[source_id][oid]
                self.object_observations[oid].append((source_id, source_observations[source_id][oid]))
                self.object_distinct_observations[oid].add(source_observations[source_id][oid])
        # initialize source accuracy --- utilize any ground truth if specified
        self.source_accuracy = {}
        self._init_src_accuracy()

    def _init_src_accuracy(self):
        for source_id in self.source_observations:
            correct = 0.0
            total = 0.0
            for oid in self.source_observations[source_id]:
                if oid in self.object_truth:
                    total += 1.0
                    self.object_inferred_truth[oid] = self.object_truth[oid]
                    if self.object_truth[oid] == self.source_observations[source_id][oid]:
                        correct += 1.0
            if total == 0.0:
                self.source_accuracy[source_id] = round(random.uniform(0.5, 1), 3)
            else:
                self.source_accuracy[source_id] = correct / total
            if self.source_accuracy[source_id] == 1.0:
                self.source_accuracy[source_id] = 0.99
            elif self.source_accuracy[source_id] == 0.0:
                self.source_accuracy[source_id] = 0.01
        return

    def update_object_assignment(self):
        for oid in self.object_observations:
            obs_scores = {}
            for (src_id, value) in self.object_observations[oid]:
                if value not in obs_scores:
                    obs_scores[value] = 0.0
                if len(self.object_distinct_observations[oid]) == 1:
                    obs_scores[value] = 1.0
                else:
                    obs_scores[value] += math.log(
                        (len(self.object_distinct_observations[oid]) - 1) * self.source_accuracy[src_id] / (
                                    1 - self.source_accuracy[src_id]))

            # assign largest score
            max_value = -1
            max_index = 0
            for i in obs_scores:
                if obs_scores[i] > max_value:
                    max_value = obs_scores[i]
                    max_index = i
            self.object_inferred_truth[oid] = max_index

    def update_source_accuracy(self):
        for source_id in self.source_observations:
            correct = 0.0
            total = 0.0
            for oid in self.source_observations[source_id]:
                if oid in self.object_inferred_truth:
                    total += 1.0
                    if self.object_inferred_truth[oid] == self.source_observations[source_id][oid]:
                        correct += 1.0
        assert total != 0.0
        self.source_accuracy[source_id] = correct / total
        if self.source_accuracy[source_id] == 1.0:
            self.source_accuracy[source_id] = 0.99
        elif self.source_accuracy[source_id] == 0.0:
            self.source_accuracy[source_id] = 0.01

    def solve(self, iterations=100):
        for i in tqdm(range(iterations)):
            self.update_object_assignment()
            self.update_source_accuracy()
        return

class Slimfast:
    def __init__(self, object_truth, source_observations, source_features,
                 alpha=0.01, reg=0.01):
        # load data
        self.object_truth = object_truth
        self.source_observations = source_observations
        self.source_features = source_features
        self.alpha = alpha
        self.reg = reg

        # init local dictionaries
        self.object_inferred_truth = {}
        self.object_distinct_observations = {}
        self.object_observations = {}
        # object to value dictionary
        for source_id in self.source_observations:
            for oid in self.source_observations[source_id]:
                if oid in object_truth:
                    continue
                if oid not in self.object_observations:
                    self.object_observations[oid] = []
                    self.object_distinct_observations[oid] = set([])
                    self.object_inferred_truth[oid] = \
                    source_observations[source_id][oid]
                self.object_observations[oid].append(
                    (source_id, source_observations[source_id][oid]))
                self.object_distinct_observations[oid].add(
                    source_observations[source_id][oid])

        # initialize source accuracy --- utilize any ground truth if specified
        self.source_accuracy = {}
        self.feature_weights = {}
        self._init_feature_weights()

        # feature counts
        self.feature_count = {}
        for oid in self.object_observations:
            for (src_id, value) in self.object_observations[oid]:
                for feat in self.source_features[src_id]:
                    if feat not in self.feature_count:
                        self.feature_count[feat] = 0.0
                    self.feature_count[feat] += 1.0

    def _init_feature_weights(self):
        for sid in self.source_features:
            for feat in self.source_features[sid]:
                self.feature_weights[feat] = 0.0
            self.source_accuracy[sid] = round(random.uniform(0.7, 0.99), 3)
        return

    def update_feature_weights(self, sid, correct):
        total_weight = 0.0
        for feat in self.source_features[sid]:
            total_weight += self.feature_weights[feat]
        for feat in self.source_features[sid]:
            if correct:
                self.feature_weights[feat] -= self.alpha * (
                        -1.0 / (math.exp(-1.0 * total_weight) + 1.0))
            else:
                self.feature_weights[feat] -= self.alpha * (
                        1.0 / (1.0 + math.exp(-1.0 * total_weight)))

    def update_object_assignment(self):
        for oid in self.object_observations:
            obs_scores = {}
            for (src_id, value) in self.object_observations[oid]:
                if value not in obs_scores:
                    obs_scores[value] = 0.0
                if len(self.object_distinct_observations[oid]) == 1:
                    obs_scores[value] = 1.0
                else:
                    obs_scores[value] += math.log(
                        (len(self.object_distinct_observations[oid]) - 1) *
                        self.source_accuracy[src_id] / (
                                1 - self.source_accuracy[src_id]))

            # assign largest score
            self.object_inferred_truth[oid] = max(obs_scores,
                                                  key=obs_scores.get)

    def update_source_accuracy(self, object_truth):
        for oid in self.object_observations:
            if oid in object_truth:
                for sid, value in self.object_observations[oid]:
                    if object_truth[oid] == value:
                        self.update_feature_weights(sid, True)
                    else:
                        self.update_feature_weights(sid, False)
        for feat in self.feature_weights:
            # L1-regularization
            if self.feature_weights[feat] > 0:
                self.feature_weights[feat] = max(0, self.feature_weights[
                    feat] - self.alpha * self.reg)
            elif self.feature_weights[feat] < 0:
                self.feature_weights[feat] = min(0, self.feature_weights[
                    feat] + self.alpha * self.reg)
        for sid in self.source_observations:
            total_weight = 0.0
            for feat in self.source_features[sid]:
                total_weight += self.feature_weights[feat]
            self.source_accuracy[sid] = 1.0 / (
                    1.0 + math.exp(-total_weight))
            if self.source_accuracy[sid] == 1.0:
                self.source_accuracy[sid] = 0.99
            elif self.source_accuracy[sid] == 0.0:
                self.source_accuracy[sid] = 0.01

    def solve(self, iterations=10):
        if len(self.object_truth) > 0:
            self.update_source_accuracy(self.object_truth)
        for i in range(iterations):
            self.update_object_assignment()
            self.update_source_accuracy(self.object_inferred_truth)

def print_source_accuracy(method):
    """
    Print the sources accuracies
    :param method: mehtod that we used for fusion
    """
    for source in method.source_accuracy:
        print (source + ","+ str(method.source_accuracy[source]))



