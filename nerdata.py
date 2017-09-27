# nerdata.py


# Abstraction to bundle words with POS and chunks for featurization
class Token:
    def __init__(self, word, pos, chunk):
        self.word = word
        self.pos = pos
        self.chunk = chunk

    def __repr__(self):
        return self.word


# Thin wrapper around a start and end index coupled with a label, representing,
# e.g., a chunk PER over the span (3,5). Indices are semi-inclusive, so (3,5)
# contains tokens 3 and 4 (0-based indexing).
class Chunk:
    def __init__(self, start_idx, end_idx, label):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label

    def __repr__(self):
        return "(" + repr(self.start_idx) + ", " + repr(self.end_idx) + ", " + self.label + ")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start_idx == other.start_idx and self.end_idx == other.end_idx and self.label == other.label
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.start_idx) + hash(self.end_idx) + hash(self.label)


# Thin wrapper over a sequence of Tokens representing a sentence and an optional set of chunks
# representation NER labels, which are also stored as BIO tags
class LabeledSentence:
    def __init__(self, tokens, chunks=None):
        self.tokens = tokens
        self.chunks = chunks
        if chunks is None:
            self.bio_tags = None
        else:
            self.bio_tags = bio_tags_from_chunks(self.chunks, len(self.tokens))

    def __repr__(self):
        return repr([repr(tok) for tok in self.tokens]) + "\n" + repr([repr(chunk) for chunk in self.chunks])

    def __len__(self):
        return len(self.tokens)

    def get_bio_tags(self):
        return self.bio_tags


# We store NER tags as strings, but they contain two pieces:
# a coarse tag type (BIO) and a label (PER), e.g. B-PER
def isB(ner_tag):
    return ner_tag.startswith("B")


def isI(ner_tag):
    return ner_tag.startswith("I")


def isO(ner_tag):
    return ner_tag == "O"


# Gets the label component of the NER tag: e.g., returns PER for B-PER
def get_tag_label(ner_tag):
    if len(ner_tag) > 2:
        return ner_tag[2:]
    else:
        return None


# Convert BIO tags to (start, end, label) chunk representations
# (start, end) are semi-inclusive, meaning that in the sentence
# He met Barack Obama yesterday
# Barack Obama has the span (2, 4)
# N.B. this method only works because chunks are non-overlapping in this data
def chunks_from_bio_tag_seq(bio_tags):
    chunks = []
    curr_tok_start = -1
    curr_tok_label = ""
    for idx, tag in enumerate(bio_tags):
        if isB(tag):
            label = get_tag_label(tag)
            if curr_tok_label != "":
                chunks.append(Chunk(curr_tok_start, idx, curr_tok_label))
            curr_tok_label = label
            curr_tok_start = idx
        elif isI(tag):
            label = get_tag_label(tag)
            # if label != curr_tok_label:
                # print "WARNING: invalid tag sequence (I after O); ignoring the I: " + repr(bio_tags)
        else: # isO(tag):
            if curr_tok_label != "":
                chunks.append(Chunk(curr_tok_start, idx, curr_tok_label))
            curr_tok_label = ""
            curr_tok_start = -1
    return chunks


# Converts a chunk representation back to BIO tags
def bio_tags_from_chunks(chunks, sent_len):
    tags = []
    for i in xrange(0, sent_len):
        matching_chunks = filter(lambda chunk: chunk.start_idx <= i and i < chunk.end_idx, chunks)
        if len(matching_chunks) > 0:
            if i == matching_chunks[0].start_idx:
                tags.append("B-" + matching_chunks[0].label)
            else:
                tags.append("I-" + matching_chunks[0].label)
        else:
            tags.append("O")
    return tags


# Reads a dataset in the CoNLL format from a file
# The format is one token per line:
# [word] [POS] [syntactic chunk] *potential junk column* [NER tag]
# One blank line appears after each sentence
def read_data(file):
    f = open(file)
    sentences = []
    curr_tokens = []
    curr_bio_tags = []
    for line in f:
        stripped = line.strip()
        if stripped != "":
            fields = stripped.split(" ")
            if len(fields) == 4 or len(fields) == 5:
                # TODO: Modify this line to remember POS tags (fields[1]) or chunks (fields[2]) if desired
                curr_tokens.append(Token(fields[0], fields[1], fields[2]))
                # N.B. fields[-1] because there are weird extra fields in .train and .testa
                curr_bio_tags.append(fields[-1])
        elif stripped == "" and len(curr_tokens) > 0:
            sentences.append(LabeledSentence(curr_tokens, chunks_from_bio_tag_seq(curr_bio_tags)))
            curr_tokens = []
            curr_bio_tags = []
    return sentences


# Evaluates the guess sentences with respect to the gold sentences
def print_evaluation(gold_sentences, guess_sentences):
    correct = 0
    num_pred = 0
    num_gold = 0
    for gold, guess in zip(gold_sentences, guess_sentences):
        correct += len(set(guess.chunks) & set(gold.chunks))
        num_pred += len(guess.chunks)
        num_gold += len(gold.chunks)
    if num_pred == 0:
        prec = 0
    else:
        prec = correct/float(num_pred)
    if num_gold == 0:
        rec = 0
    else:
        rec = correct/float(num_gold)
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    print "Labeled F1: " + "{0:.2f}".format(f1 * 100) +\
          ", precision: " + repr(correct) + "/" + repr(num_pred) + " = " + "{0:.2f}".format(prec * 100) + \
          ", recall: " + repr(correct) + "/" + repr(num_gold) + " = " + "{0:.2f}".format(rec * 100)


# Writes labeled_sentences to outfile in the CoNLL format
def print_output(labeled_sentences, outfile):
    f = open(outfile, 'w')
    for sentence in labeled_sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tok = sentence.tokens[i]
            f.write(tok.word + " " + tok.pos + " " + tok.chunk + " " + bio_tags[i] + "\n")
        f.write("\n")
    f.close()
