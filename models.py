# models.py

from nerdata import *
from utils import *
from adagrad_trainer import *

import numpy as np
import scipy.misc

# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        score = np.zeros((len(sentence), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence), len(self.tag_indexer))) * -1
        sequence_scorer = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        for idx in xrange(0, len(sentence)):
            # Initial Scores
            if idx == 0:
                for tag_idx in xrange(0, len(self.tag_indexer)):    
                    score[idx][tag_idx] = sequence_scorer.score_init(sentence, tag_idx) + \
                                            sequence_scorer.score_emission(sentence, tag_idx, idx)
            else:
                for curr_tag_idx in xrange(0, len(self.tag_indexer)):
                    score[idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in xrange(0, len(self.tag_indexer)):
                        curr_score = sequence_scorer.score_transition(sentence, prev_tag_idx, curr_tag_idx) + \
                                        sequence_scorer.score_emission(sentence, curr_tag_idx, idx) + score[idx-1][prev_tag_idx]
                        if curr_score > score[idx][curr_tag_idx]:
                            score[idx][curr_tag_idx] = curr_score
                            back_pointers[idx][curr_tag_idx] = prev_tag_idx
        max_score_idx = score.argmax(axis=1)[-1]
        idx = max_score_idx
        pred_tags = []
        count = len(sentence) - 1
        while idx != -1 :
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = back_pointers[count][idx]
            count = count - 1
        pred_tags.reverse()
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print repr(init_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print "Tag indexer: " + repr(tag_indexer)
    print "Initial state log probabilities: " + repr(init_counts)
    print "Transition log probabilities: " + repr(transition_counts)
    print "Emission log probs too big to print..."
    print "Emission log probs for India: " + repr(emission_counts[:,word_indexer.get_index("India")])
    print "Emission log probs for Phil: " + repr(emission_counts[:,word_indexer.get_index("Phil")])
    print "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)"
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)

# Implement the forward pass of the algorithm
# scored_feature is the phi function(y_t, y_(t-1), x_t)
def forward_pass(sentence, tag_indexer, scored_feature):
    alpha = np.zeros((len(sentence), len(tag_indexer)))
    for tag_idx in xrange(0, len(tag_indexer)):
        alpha[0][tag_idx] = scored_feature[0][tag_idx][0]
    for word_idx in xrange(0, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            for prev_tag_idx in xrange(0, len(tag_indexer)):
                alpha[word_idx][tag_idx] += alpha[word_idx - 1][prev_tag_idx] * scored_feature[tag_idx][prev_tag_idx][word_idx]
    return alpha

# Implement the backward pass of the algorithm
# scored_feature is the phi function(y_t, y_(t-1), x_t)
def backward_pass(sentence, tag_indexer, scored_feature):
    beta = np.zeros((len(sentence), len(tag_indexer)))
    for tag_idx in xrange(0, len(tag_indexer)):
        beta[len(sentence)-1][tag_idx] = 1
    for word_idx in range(len(sentence)-1, -1, -1):
        for tag_idx in range(0, len(tag_indexer)):
            for next_tag_idx in range(0, len(tag_indexer)):
                beta[word_idx][tag_idx] += beta[word_idx + 1][next_tag_idx] * scored_feature[next_tag_idx][tag_idx][word_idx]
    return beta


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)


class FeatureBasedSequenceScorer(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def score_init(self, feature_cache, tag_idx):
        return score_indexed_features(feature_cache[0][tag_idx], self.feature_weights)

    def score_transition(self, feature_cache, prev_tag_idx, curr_tag_idx):
        return 0

    def score_emission(self, feature_cache, tag_idx, word_idx):
        return score_indexed_features(feature_cache[word_idx][tag_idx], self.feature_weights)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        feature_cache = [[[] for k in xrange(0, len(self.tag_indexer))] for j in xrange(0, len(sentence))]
        for word_idx in range(0, len(sentence)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)

        # Viterbi
        score = np.zeros((len(sentence), len(self.tag_indexer)))
        back_pointers = np.ones((len(sentence), len(self.tag_indexer))) * -1
        sequence_scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights)
        for word_idx in xrange(0, len(sentence)):
            if word_idx == 0:
                for tag_idx in xrange(0, len(self.tag_indexer)):
                    tag = self.tag_indexer.get_object(tag_idx)
                    if isI(tag):
                        score[word_idx][tag_idx] = -np.inf
                    else:    
                        score[word_idx][tag_idx] = sequence_scorer.score_init(feature_cache, tag_idx)
            else:
                for curr_tag_idx in xrange(0, len(self.tag_indexer)):
                    score[word_idx][curr_tag_idx] = -np.inf
                    for prev_tag_idx in xrange(0, len(self.tag_indexer)):
                        # TODO : did not prohibit the O-I transition at the last word
                        curr_tag = self.tag_indexer.get_object(curr_tag_idx)
                        prev_tag = self.tag_indexer.get_object(prev_tag_idx)
                        if isO(prev_tag) and isI(curr_tag):
                            continue
                        if isI(curr_tag) and (get_tag_label(curr_tag) != get_tag_label(prev_tag)):
                            continue
                        curr_score = sequence_scorer.score_transition(feature_cache, prev_tag_idx, curr_tag_idx) + \
                                        sequence_scorer.score_emission(feature_cache, curr_tag_idx, word_idx) + score[word_idx-1][prev_tag_idx]
                        if curr_score > score[word_idx][curr_tag_idx]:
                            score[word_idx][curr_tag_idx] = curr_score
                            back_pointers[word_idx][curr_tag_idx] = prev_tag_idx
        max_score_idx = score.argmax(axis=1)[-1]
        idx = max_score_idx
        pred_tags = []
        word_idx = len(sentence) - 1
        while idx != -1 :
            pred_tags.append(self.tag_indexer.get_object(idx))
            idx = back_pointers[word_idx][idx]
            word_idx -= 1
        pred_tags.reverse()
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences, epochs, lr, weights_file="", output_weights=""):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    transition_mat = np.ones((len(tag_indexer), len(tag_indexer)))
    for tag_idxa in range(0, len(tag_indexer)):
        for tag_idxb in range(0, len(tag_indexer)):
            tag_a = tag_indexer.get_object(tag_idxa)
            tag_b = tag_indexer.get_object(tag_idxb)
            if isI(tag_b) and (get_tag_label(tag_b) != get_tag_label(tag_a)):
                transition_mat[tag_idxa][tag_idxb] = 0
    print "Extracting features"
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]
    for sentence_idx in xrange(0, len(sentences)):
        if sentence_idx % 500 == 0:
            print("Ex " + repr(sentence_idx) + "/" + repr(len(sentences)))
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            for tag_idx in xrange(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    feature_weights = np.random.rand((len(feature_indexer)))
    if weights_file != "":
        feature_weights = np.load(weights_file)

    print("Initital Statistics")
    model = CrfNerModel(tag_indexer, feature_indexer, feature_weights)
    # TODO : currently using only emission features, also extend to transition features if possible
    batch_size = 1
    # training loop
    for epoch in range(0, epochs):
        print("Epoch %d" % (epoch+1))
        gradient = Counter()
        for sentence_idx in range(0, len(sentences)):
            if sentence_idx%500 == 0:
                print('Training on ' + repr(sentence_idx))
            log_marginal_probs = compute_log_marginals(sentences[sentence_idx], tag_indexer, feature_cache[sentence_idx], model.feature_weights)

            for word_idx in range(0, len(sentences[sentence_idx])):
                for tag_idx in range(0, len(tag_indexer)):
                    gradient.increment_all(feature_cache[sentence_idx][word_idx][tag_idx], - np.exp(log_marginal_probs[word_idx][tag_idx]))
                gold_tag = sentences[sentence_idx].get_bio_tags()[word_idx]
                gold_tag_idx = tag_indexer.index_of(gold_tag)
                gradient.increment_all(feature_cache[sentence_idx][word_idx][gold_tag_idx], 1.0)
            if (sentence_idx+1) % batch_size == 0:
                for weight_idx in gradient.keys():
                    model.feature_weights[weight_idx] += (lr * gradient.get_count(weight_idx))/batch_size
                gradient = Counter()
    np.save(output_weights, model.feature_weights)
    return model

# TODO : implementation specific to emission features only, change forward and backward if adding transition features
def compute_log_marginals(sentence, tag_indexer, sentence_feature_cache, feature_weights):
    # find alpha -> forward pass
    log_alpha = np.zeros((len(sentence), len(tag_indexer)))
    for tag_idx in xrange(0, len(tag_indexer)):
        log_alpha[0][tag_idx] = score_indexed_features(sentence_feature_cache[0][tag_idx], feature_weights)
    for word_idx in xrange(1, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            log_alpha[word_idx][tag_idx] = -np.inf
            for prev_tag_idx in xrange(0, len(tag_indexer)):
                curr_tag = tag_indexer.get_object(tag_idx)
                prev_tag = tag_indexer.get_object(prev_tag_idx)
                if isI(curr_tag) and get_tag_label(curr_tag) != get_tag_label(prev_tag):
                    continue
                log_alpha[word_idx][tag_idx] = np.logaddexp(log_alpha[word_idx][tag_idx], \
                                                            log_alpha[word_idx - 1][prev_tag_idx] + \
                                                        score_indexed_features(sentence_feature_cache[word_idx][tag_idx], feature_weights))
             
            # log_alpha[word_idx][tag_idx] = scipy.misc.logsumexp(log_alpha[word_idx-1] + score_indexed_features(sentence_feature_cache[word_idx][tag_idx], feature_weights))
    
    # find beta -> backward pass
    log_beta = np.zeros((len(sentence), len(tag_indexer)))
    for word_idx in range(len(sentence)-2, -1, -1):
        for tag_idx in range(0, len(tag_indexer)):
            log_beta[word_idx][tag_idx] = -np.inf
            for next_tag_idx in range(0, len(tag_indexer)):
                curr_tag = tag_indexer.get_object(tag_idx)
                next_tag = tag_indexer.get_object(next_tag_idx)
                if isI(next_tag) and get_tag_label(curr_tag) != get_tag_label(next_tag):
                    continue
                log_beta[word_idx][tag_idx] = np.logaddexp(log_beta[word_idx][tag_idx], \
                                                        log_beta[word_idx + 1][next_tag_idx] + \
                                                        score_indexed_features(sentence_feature_cache[word_idx][next_tag_idx], feature_weights))
            # tmp = np.apply_along_axis(score_indexed_features, 1, sentence_feature_cache[word_idx], feature_weights)
            # log_beta[word_idx][tag_idx] = scipy.misc.logsumexp(log_beta[word_idx + 1] + tmp)

    # marginal = alpha[word_idx][tag_idx] * beta[word_idx][tag_idx] / Sigma (alpha, beta)
    log_marginal_probs = np.zeros((len(sentence), len(tag_indexer)))
    log_marginal_probs = log_alpha + log_beta
    # denom = np.apply_along_axis(scipy.misc.logsumexp, 1, log_marginal_probs)
    # log_marginal_probs -= denom[:, None]
    for word_idx in range(0, len(sentence)):
        denom = -np.inf
        for tag_idx in range(0, len(tag_indexer)):
            denom = np.logaddexp(denom, log_marginal_probs[word_idx][tag_idx])
        log_marginal_probs[word_idx] -= denom
    return log_marginal_probs

# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
