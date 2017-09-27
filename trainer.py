# trainer.py

import sys
from nerdata import *
from utils import *
from models import *
import argparse

class BadNerModel():
    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence):
        pred_tags = []
        for tok in sentence.tokens:
            if self.words_to_tag_counters.has_key(tok.word):
                pred_tags.append(self.words_to_tag_counters[tok.word].argmax())
            else:
                pred_tags.append("O")
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def train_bad_ner_model(training_set):
    words_to_tag_counters = {}
    for sentence in training_set:
        tags = sentence.get_bio_tags()
        for idx in xrange(0, len(sentence)):
            word = sentence.tokens[idx].word
            if not words_to_tag_counters.has_key(word):
                words_to_tag_counters[word] = Counter()
                words_to_tag_counters[word].increment_count(tags[idx], 1.0)
    return BadNerModel(words_to_tag_counters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NER tagging system')
    parser.add_argument('--model', dest='model', type=str, default='BAD')
    parser.add_argument('--preW', dest='preW', type=str, default='')
    parser.add_argument('--outW', dest='outW', type=str, default='')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('-l', '--language', dest='lang', type=str, default='eng')
    parser.add_argument('-lr', '--lr', dest='lr', type=float, default=0.1)
    args = parser.parse_args()

    # Load the training and test data
    train = read_data("data/" + args.lang + ".train")
    dev = read_data("data/" + args.lang + ".testa")
    # Here's a few sentences...
    # print "Examples of sentences:"
    # print str(dev[1])
    # print str(dev[3])
    # print str(dev[5])
    system_to_run = args.model
    # Set to True when you're ready to run your CRF on the test set to produce the final output
    run_on_test = True
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_ner_model(train)
        dev_decoded = [bad_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "HMM":
        hmm_model = train_hmm_model(train)
        dev_decoded = [hmm_model.decode(test_ex) for test_ex in dev]
    elif system_to_run == "CRF":
        crf_model = train_crf_model(train, args.epochs, args.lr, weights_file=args.preW, output_weights=args.outW)
        dev_decoded = [crf_model.decode(test_ex) for test_ex in dev]
        if run_on_test:
            test = read_data("data/eng.testb.blind")
            test_decoded = [crf_model.decode(test_ex) for test_ex in test]
            print_output(test_decoded, "eng.testb.out")
    else:
        raise Exception("Pass in either BAD, HMM, or CRF to run the appropriate system")
    # Print the evaluation statistics
    print_evaluation(dev, dev_decoded)
