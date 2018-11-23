import tensorflow as tf
import re
from collections import Counter
import string
import math
import numpy as np
from prepro import word_tokenize
from meteor import Meteor

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        ans_limit = config.test_ans_limit if is_test else config.ans_limit
        char_limit = config.char_limit

        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_idxs_ans": tf.FixedLenFeature([], tf.string),
                                               "context_idxs_ans_unique": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs_ans": tf.FixedLenFeature([], tf.string),
                                               "ans_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs_ans": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ans_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })

        context_idxs = tf.reshape(tf.decode_raw(
                features["context_idxs"], tf.int32), [para_limit])
        context_idxs_ans = tf.reshape(tf.decode_raw(
                features["context_idxs_ans"], tf.int32), [para_limit])
        context_idxs_ans_unique = tf.reshape(tf.decode_raw(
                features["context_idxs_ans_unique"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
                features["ques_idxs"], tf.int32), [ques_limit])
        ques_idxs_ans = tf.reshape(tf.decode_raw(
                features["ques_idxs_ans"], tf.int32), [ques_limit])
        ans_idxs = tf.reshape(tf.decode_raw(
                features["ans_idxs"], tf.int32), [ans_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(
                features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        context_char_idxs_ans = tf.reshape(tf.decode_raw(
                features["context_char_idxs_ans"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(
                features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        ans_char_idxs = tf.reshape(tf.decode_raw(
                features["ans_char_idxs"], tf.int32), [ans_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(
                features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
                features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]

        return context_idxs, context_idxs_ans, context_idxs_ans_unique, ques_idxs, ques_idxs_ans, ans_idxs, context_char_idxs, \
               context_char_idxs_ans, ques_char_idxs, ans_char_idxs, y1, y2, qa_id

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
        context_tokens = eval_file[str(qid)]["context_tokens_ans"]
        if 3 in syms:
            syms = syms[:syms.index(3)]
        answer = u' '.join([id2word[sym] if sym in id2word
                            else context_tokens[sym - len(id2word)] for sym in syms])
        answer_dict[str(qid)] = answer
        remapped_dict[uuid] = answer
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict, is_answer=True):
    f1 = exact_match = total = 0
    f1s = {}
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["questions"] if not is_answer else eval_file[key]["answers"]
        prediction = value
        em = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths, is_answer=is_answer)
        exact_match += em
        f_1 = metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths, is_answer=is_answer)
        f1 += f_1
        f1s[key] = str(f_1)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}, f1s


def evaluate_rl(eval_file, qa_id, symbols, symbols_rl, id2word, baseline_file=None, is_answer=True,
                metric="f1", has_baseline=True, if_fix_base=False):
    rewards = []
    rewards_rl = []
    rewards_base = []
    res_meteor, res_meteor_rl, gts_meteor = [], [], []
    for qid, syms, syms_rl in zip(qa_id, zip(*symbols), zip(*symbols_rl)):
        ground_truths = eval_file[str(qid)]["questions"] if not is_answer else eval_file[str(qid)]["answers"]
        context_tokens = eval_file[str(qid)]["context_tokens_ans"]
        if 3 in syms:
            syms = syms[:syms.index(3)]
        answer = u' '.join([id2word[sym] if sym in id2word
                            else context_tokens[sym - len(id2word)] for sym in syms])
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        answer_rl = u' '.join([id2word[sym_rl] if sym_rl in id2word
                               else context_tokens[sym_rl - len(id2word)] for sym_rl in syms_rl])
        if metric == "f1":
            f1 = metric_max_over_ground_truths(f1_score, answer, ground_truths) if not if_fix_base \
                else float(baseline_file[str(qid)])
            f1_rl = metric_max_over_ground_truths(f1_score, answer_rl, ground_truths)
            rewards.append(f1_rl - f1 if has_baseline else f1_rl)
            rewards_rl.append(f1_rl)
            rewards_base.append(f1)
        elif metric == "bleu":
            answer = normalize_answer(answer).split()
            answer_rl = normalize_answer(answer_rl).split()
            ground_truths = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]
            bleu = compute_bleu([ground_truths], [answer])[0] if not if_fix_base \
                else float(baseline_file[str(qid)])
            bleu_rl = compute_bleu([ground_truths], [answer_rl])[0]
            rewards.append(bleu_rl - bleu if has_baseline else bleu_rl)
            rewards_rl.append(bleu_rl)
            rewards_base.append(bleu)
        elif metric == "rouge":
            answer = normalize_answer(answer).split()
            answer_rl = normalize_answer(answer_rl).split()
            ground_truths = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]
            rouge = compute_rouge_L(answer, ground_truths) if not if_fix_base \
                else float(baseline_file[str(qid)])
            rouge_rl = compute_rouge_L(answer_rl, ground_truths)
            rewards.append(rouge_rl - rouge if has_baseline else rouge_rl)
            rewards_rl.append(rouge_rl)
            rewards_base.append(rouge)
        elif metric == "meteor":
            gts_meteor.append([normalize_answer(ground_truth) for ground_truth in ground_truths])
            res_meteor.append(normalize_answer(answer))
            res_meteor_rl.append(normalize_answer(answer_rl))
    if metric == "meteor":
        meteor = Meteor()
        rewards_base = meteor.compute_score(gts_meteor, res_meteor)[1]
        rewards_rl = meteor.compute_score(gts_meteor, res_meteor_rl)[1]
        rewards = map(lambda x: x[0] - x[1], zip(rewards_rl, rewards_base)) if has_baseline else rewards_rl
    return np.array(rewards), np.mean(rewards_rl), np.mean(rewards_base)


def evaluate_rl_dual(eval_file, qa_id, base_dual_byp1, base_dual_byp2, dual_byp1, dual_byp2,
                     base_dual_loss, dual_loss, dual_rl_metric, has_baseline):
    reward = []
    reward_rl = []
    reward_base = []
    if dual_rl_metric == "prob":
        reward_base = map(lambda x: np.exp(-x), list(base_dual_loss))
        reward_rl = map(lambda x: np.exp(-x), list(dual_loss))
        reward = map(lambda x: x[0] - x[1], zip(reward_rl, reward_base)) if has_baseline else reward_rl
    elif dual_rl_metric == "f1":
        base_dual_yp1 = map(lambda x: x[0], base_dual_byp1)
        base_dual_yp2 = map(lambda x: x[0], base_dual_byp2)
        dual_yp1 = map(lambda x: x[0], dual_byp1)
        dual_yp2 = map(lambda x: x[0], dual_byp2)
        for qid, base_p1, base_p2, p1, p2 in zip(qa_id, base_dual_yp1, base_dual_yp2, dual_yp1, dual_yp2):
            ground_truths = eval_file[str(qid)]["answers"]
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            base_start_idx = spans[base_p1][0]
            base_end_idx = spans[base_p2][1]
            base_answer = context[base_start_idx: base_end_idx]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer = context[start_idx: end_idx]
            base_f1 = metric_max_over_ground_truths(f1_score, base_answer, ground_truths)
            f1 = metric_max_over_ground_truths(f1_score, answer, ground_truths)
            reward_base.append(base_f1)
            reward_rl.append(f1)
            reward.append(f1 - base_f1 if has_baseline else f1)
    return np.array(reward), np.mean(reward_rl), np.mean(reward_base)


def format_generated_questions(eval_file, qa_id, symbols, symbols_rl, batch_size, ques_limit, char_limit, id2word, char2idx_dict):
    ques_idxs, ques_idxs_rl = np.zeros([batch_size, ques_limit], dtype=np.int32), np.zeros([batch_size, ques_limit], dtype=np.int32)
    ques_char_idxs, ques_char_idxs_rl = np.zeros([batch_size, ques_limit, char_limit], dtype=np.int32), \
                                        np.zeros([batch_size, ques_limit, char_limit], dtype=np.int32)
    for k, (qid, syms, syms_rl) in enumerate(zip(qa_id, zip(*symbols), zip(*symbols_rl))):
        context_tokens = eval_file[str(qid)]["context_tokens_ans"]
        syms, syms_rl = list(syms), list(syms_rl)
        if 3 in syms:
            syms = syms[:syms.index(3)]
        syms = [2] + syms[:ques_limit - 2] + [3]
        for i, sym in enumerate(syms):
            ques_idxs[k, i] = sym
            word = id2word[sym] if sym in id2word else context_tokens[sym - len(id2word)]
            for j, c in enumerate(list(word)):
                if j == char_limit:
                    break
                ques_char_idxs[k, i, j] = char2idx_dict[c] if c in char2idx_dict else 1
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        syms_rl = [2] + syms_rl[:ques_limit - 2] + [3]
        for i, sym_rl in enumerate(syms_rl):
            ques_idxs_rl[k, i] = sym_rl
            word = id2word[sym_rl] if sym_rl in id2word else context_tokens[sym_rl - len(id2word)]
            for j, c in enumerate(list(word)):
                if j == char_limit:
                    break
                ques_char_idxs_rl[k, i, j] = char2idx_dict[c] if c in char2idx_dict else 1
    return ques_idxs, ques_char_idxs, ques_idxs_rl, ques_char_idxs_rl


def format_sampled_questions(symbols_rl, batch_size, ques_limit):
    ques_idxs_rl = np.zeros([batch_size, ques_limit], dtype=np.int32)
    for k, syms_rl in enumerate(zip(*symbols_rl)):
        syms_rl = list(syms_rl)
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        # add GO and EOS in question
        syms_rl = [2] + syms_rl[:ques_limit - 2] + [3]
        for i, sym_rl in enumerate(syms_rl):
            ques_idxs_rl[k, i] = sym_rl
    return ques_idxs_rl


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


def f1_score(prediction, ground_truth, is_answer=True):
    prediction_tokens = normalize_answer(prediction).split() if is_answer else prediction.lower().split()
    ground_truth_tokens = normalize_answer(ground_truth).split() if is_answer else word_tokenize(ground_truth.lower())
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, is_answer=True):
    if is_answer:
        return (normalize_answer(prediction) == normalize_answer(ground_truth))
    else:
        return (' '.join(prediction.lower().split()) == ' '.join(word_tokenize(ground_truth.lower())))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, is_answer=True):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, is_answer=is_answer)
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
        bp = math.exp(1 - 1. / (ratio + 1e-12))

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def evaluate_bleu(eval_file, answer_dict, is_answer=True):
    reference_corpus = []
    translation_corpus = []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"] if is_answer else eval_file[key]["questions"]
        # prediction_tokens = normalize_answer(prediction).split()
        prediction_tokens = value.lower().split()
        # ground_truth_tokens = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]
        ground_truth_tokens = [word_tokenize(ground_truth.lower()) for ground_truth in ground_truths]
        translation_corpus.append(prediction_tokens)
        reference_corpus.append(ground_truth_tokens)
    return compute_bleu(reference_corpus, translation_corpus)


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


def compute_rouge_L(pred, refs, beta = 1.2):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    prec = []
    rec = []
    for ref in refs:
        # compute the longest common subsequence
        lcs = my_lcs(pred, ref)
        prec.append(lcs / float(len(pred)) if len(pred) != 0 else 0.0)
        rec.append(lcs / float(len(ref)) if len(ref) != 0 else 0.0)

    prec_max = max(prec)
    rec_max = max(rec)

    if prec_max != 0 and rec_max != 0:
        score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
    else:
        score = 0.0
    return score


def evaluate_rouge_L(eval_file, answer_dict, is_answer=True):
    scores = []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"] if is_answer else eval_file[key]["questions"]
        # prediction = value
        # prediction_tokens = normalize_answer(prediction).split()
        prediction_tokens = value.lower().split()
        # ground_truth_tokens = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]
        ground_truth_tokens = [word_tokenize(ground_truth.lower()) for ground_truth in ground_truths]
        score = compute_rouge_L(prediction_tokens, ground_truth_tokens)
        scores.append(score)
    return np.mean(scores)


def evaluate_meteor(eval_file, answer_dict, is_answer=True):
    meteor = Meteor()
    res, gts = [], []
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"] if is_answer else eval_file[key]["questions"]
        # prediction = normalize_answer(value)
        prediction = ' '.join(value.lower().split())
        ground_truths = [' '.join(word_tokenize(ground_truth.lower())) for ground_truth in ground_truths]
        res.append(prediction)
        gts.append(ground_truths)
    score = meteor.compute_score(gts, res)
    return score


if __name__ == '__main__':
    a = ["hello", ",", "i", "love", "you"]
    b = [["hello", ",", "i", "love", "you", "!"]]
    print compute_rouge_L(a, b)
