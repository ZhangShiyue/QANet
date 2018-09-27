import tensorflow as tf
import re
from collections import Counter
import string
import math
import numpy as np
from prepro import word_tokenize

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, ques_limit, ans_limit, is_test=False, is_rerank=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        char_limit = config.char_limit

        if is_rerank:
            features = tf.parse_single_example(example,
                                               features={
                                                   "context_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ques_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ans_idxs": tf.FixedLenFeature([], tf.string),
                                                   "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ans_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "y1": tf.FixedLenFeature([], tf.string),
                                                   "y2": tf.FixedLenFeature([], tf.string),
                                                   "id": tf.FixedLenFeature([], tf.int64),
                                                   "cid": tf.FixedLenFeature([], tf.int64)
                                               })
        else:
            features = tf.parse_single_example(example,
                                               features={
                                                   "context_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ques_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ans_idxs": tf.FixedLenFeature([], tf.string),
                                                   "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "ans_char_idxs": tf.FixedLenFeature([], tf.string),
                                                   "y1": tf.FixedLenFeature([], tf.string),
                                                   "y2": tf.FixedLenFeature([], tf.string),
                                                   "id": tf.FixedLenFeature([], tf.int64)
                                               })

        context_idxs = tf.reshape(tf.decode_raw(
                features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
                features["ques_idxs"], tf.int32), [ques_limit])
        ans_idxs = tf.reshape(tf.decode_raw(
                features["ans_idxs"], tf.int32), [ans_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
                features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
                features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        ans_char_idxs = tf.reshape(tf.decode_raw(
                features["ans_char_idxs"], tf.int32), [ans_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
                features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
                features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        if is_rerank:
            can_id = features["cid"]
            return context_idxs, ques_idxs, ans_idxs, context_char_idxs, ques_char_idxs, ans_char_idxs, y1, y2, qa_id, can_id
        else:
            return context_idxs, ques_idxs, ans_idxs, context_char_idxs, ques_char_idxs, ans_char_idxs, y1, y2, qa_id

    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
            parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, ans_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                    tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
                key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config, is_test=False):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    batch_size = config.test_batch_size if is_test else config.batch_size
    dataset = tf.data.TFRecordDataset(record_file).map(
            parser, num_parallel_calls=num_threads).repeat().batch(batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def convert_tokens_g(eval_file, qa_id, symbols, id2word):
    answer_dict = {}
    remapped_dict = {}
    for qid, syms in zip(qa_id, zip(*symbols)):
        uuid = eval_file[str(qid)]["uuid"]
        context_tokens = eval_file[str(qid)]["context_tokens"]
        if 3 in syms:
            syms = syms[:syms.index(3)]
        answer = u' '.join([id2word[sym] if sym in id2word
                            else context_tokens[sym - len(id2word)] for sym in syms])
        answer_dict[str(qid)] = answer
        remapped_dict[uuid] = answer
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict, is_answer=True):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["questions"] if not is_answer else eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def evaluate_rl(eval_file, qa_id, symbols, symbols_rl, id2word, is_answer=True):
    rewards = []
    for qid, syms, syms_rl in zip(qa_id, zip(*symbols), zip(*symbols_rl)):
        ground_truths = eval_file[str(qid)]["questions"] if not is_answer else eval_file[str(qid)]["answers"]
        context = eval_file[str(qid)]["context"].replace(
                "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
        context_tokens = word_tokenize(context)
        if 3 in syms:
            syms = syms[:syms.index(3)]
        answer = u' '.join([id2word[sym] if sym in id2word
                            else context_tokens[sym - len(id2word)] for sym in syms])
        f1 = metric_max_over_ground_truths(f1_score, answer, ground_truths)
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        answer_rl = u' '.join([id2word[sym_rl] if sym_rl in id2word
                               else context_tokens[sym_rl - len(id2word)] for sym_rl in syms_rl])
        f1_rl = metric_max_over_ground_truths(f1_score, answer_rl, ground_truths)
        rewards.append(f1_rl - f1)
    return np.array(rewards)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def evaluate_bleu(eval_file, answer_dict, is_answer=True):
    reference_corpus = []
    translation_corpus = []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"] if is_answer else eval_file[key]["questions"]
        prediction = value
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]
        translation_corpus.append(prediction_tokens)
        reference_corpus.append(ground_truth_tokens)
    return compute_bleu(reference_corpus, translation_corpus)


if __name__ == '__main__':
    a = [["hello", ",", "i", "love", "you"]]
    b = [[["hello", ",", "i", "love", "you", "!"]]]
    print compute_bleu(b, a)