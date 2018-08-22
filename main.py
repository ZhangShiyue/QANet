import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
import re

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from prepro import word_tokenize, save


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config, len(word_mat) + config.para_limit)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, word_mat, char_mat, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 100.0
        patience = 0
        best_f1 = 0.
        best_em = 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)
            train_next_element = train_iterator.get_next()
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                c, q, a, ch, qh, ah, y1, y2, qa_id = sess.run(train_next_element)
                loss, index, logits, train_op = sess.run([model.loss, model.index, model.logits, model.train_op], feed_dict={
                    model.c: c, model.q: q, model.a: a, model.ch: ch, model.qh: qh, model.ah: ah,
                    model.y1: y1, model.y2: y2,
                    model.qa_id: qa_id, model.dropout: config.dropout})
                # print sum(enc[0][1]), max(enc[0][1]), min(enc[0][1])
                # for i in range(32):
                #     print max(logits[0][i]), min(logits[0][i])
                #     print max(gen_probs[0][i]), min(gen_probs[0][i])
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(config, model, config.val_num_batches,
                                             train_eval_file, sess, "train", train_iterator)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                                                   dev_eval_file, sess, "dev", dev_iterator)

                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(
                            config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)


def evaluate_batch(config, model, num_batches, eval_file, sess, data_type, iterator):
    answer_dict = {}
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    id2word = {word_dictionary[w]: w for w in word_dictionary}
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        c, q, a, ch, qh, ah, y1, y2, qa_id = sess.run(next_element)
        shapes = a.shape
        a = np.zeros(shapes, dtype=np.int32)
        a[:, 0] = [2] * config.batch_size
        for i in range(1, config.ans_limit):
            preds = sess.run(model.preds, feed_dict={model.c: c, model.q: q, model.a: a,
                     model.ch: ch, model.qh: qh, model.ah: ah, model.y1: y1, model.y2: y2})
            a[:, i] = preds[:, i - 1]

        for qid, symbols in zip(qa_id, a):
            context = eval_file[str(qid)]["context"].replace(
                            "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
            context_tokens = word_tokenize(context)
            symbols = list(symbols)
            if 3 in symbols:
                symbols = symbols[:symbols.index(3)]
            answer = u' '.join([id2word[symbol] if symbol in id2word
                        else context_tokens[symbol - len(id2word)] for symbol in symbols[1:]])
            # deal with special symbols like %, $ etc
            elim_pre_spas = [u' %', u" 's", u' ,']
            for s in elim_pre_spas:
                if s in answer:
                    answer = s[1:].join(answer.split(s))
            elim_beh_spas = [u'$ ', u'\xa3 ', u'# ']
            for s in elim_beh_spas:
                if s in answer:
                    answer = s[:-1].join(answer.split(s))
            elim_both_spas = [u' - ']
            for s in elim_both_spas:
                if s in answer:
                    answer = s[1:-1].join(answer.split(s))
            answer_dict_ = {str(qid): answer}
            answer_dict.update(answer_dict_)
    metrics = evaluate(eval_file, answer_dict)
    f1_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [f1_sum, em_sum]


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
                config, len(word_mat) + config.test_para_limit, is_test=True),
                                 config, is_test=True).make_one_shot_iterator()
        test_next_element = test_batch.get_next()
        model = Model(config, word_mat, char_mat, trainable=False, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            answer_dict = {}
            for step in tqdm(range(total // config.test_batch_size + 1)):
                c, q, a, ch, qh, ah, y1, y2, qa_id = sess.run(test_next_element)
                shapes = a.shape
                a = np.zeros(shapes)
                a[:, 0] = [2] * config.batch_size
                for i in range(1, config.test_ans_limit):
                    preds = sess.run(model.preds, feed_dict={model.c: c, model.q: q, model.a: a,
                         model.ch: ch, model.qh: qh, model.ah: ah, model.y1: y1, model.y2: y2})
                    a[:, i] = preds[:, i - 1]
                for qid, symbols in zip(qa_id, a):
                    symbols = list(symbols)
                    if 3 in symbols:
                        symbols = symbols[:symbols.index(3)]
                    answer = u' '.join([id2word[symbol] for symbol in symbols][1:])
                    # deal with special symbols like %, $ etc
                    elim_pre_spas = [u' %', u" 's", u' ,']
                    for s in elim_pre_spas:
                        if s in answer:
                            answer = s[1:].join(answer.split(s))
                    elim_beh_spas = [u'$ ', u'\xa3 ', u'# ']
                    for s in elim_beh_spas:
                        if s in answer:
                            answer = s[:-1].join(answer.split(s))
                    elim_both_spas = [u' - ']
                    for s in elim_both_spas:
                        if s in answer:
                            answer = s[1:-1].join(answer.split(s))
                    answer_dict_ = {str(qid): answer}
                    answer_dict.update(answer_dict_)

            metrics = evaluate(eval_file, answer_dict)
            with open("{}.json".format(config.answer_file), "w") as fh:
                json.dump(answer_dict, fh)
            print("D: Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))



def test_beam(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open("{}{}.json".format(config.res_g_b_file, config.beam_size), "r") as fh:
        g_answer_dict = json.load(fh)
    with open("{}{}.json".format(config.res_d_b_file, config.beam_size), "r") as fh:
        d_answer_dict = json.load(fh)
    answer_dict = {}
    for qa_id in d_answer_dict:
        for a in d_answer_dict[qa_id]:
            if a in g_answer_dict[qa_id]:
                answer_dict[qa_id] = a
        if qa_id not in answer_dict:
            answer_dict[qa_id] = d_answer_dict[qa_id][0]
    metrics = evaluate(eval_file, answer_dict)
    print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))


def test_rerank(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    with open("{}{}.json".format(config.res_d_b_file, config.beam_size), "r") as fh:
        d_answer_dict = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset("{}{}.{}".format(config.rerank_file.split('.')[0], config.beam_size,
                                                          config.rerank_file.split('.')[1]), get_record_parser(
                config, len(word_mat) + config.test_para_limit, is_test=True, is_rerank=True),
                                 config, is_test=True).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, rerank=True, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            scores = {}
            for step in tqdm(range(total // config.test_batch_size + 1)):
                qa_id, can_id, loss = sess.run(
                        [model.qa_id, model.can_id, model.batch_loss])
                for qid, cid, l in zip(qa_id, can_id, loss):
                    if qid not in scores:
                        scores[qid] = []
                    scores[qid].append((cid, l))
            reranked = {qid: sorted(scores[qid], key=lambda x: x[1])[0][0] for qid in scores}
            # for qid in reranked:
            #     if d_answer_dict[str(qid)][0] in eval_file[str(qid)]["answers"] and \
            #                     d_answer_dict[str(qid)][reranked[qid][0][0]] not in eval_file[str(qid)]["answers"]:
            #         for cid, l in reranked[qid]:
            #             print d_answer_dict[str(qid)][cid].encode('utf-8'), l
            #         print "groundtruth: {}".format(eval_file[str(qid)]["answers"])
            #         print
            answer_dict = {str(qid): d_answer_dict[str(qid)][reranked[qid]] for qid in reranked}
            metrics = evaluate(eval_file, answer_dict)
            # with open(config.answer_file, "w") as fh:
            #     json.dump(answer_dict, fh)
            print("Exact Match: {}, F1: {}".format(
                    metrics['exact_match'], metrics['f1']))


def tmp(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.answer_file, "r") as fh:
        answer = json.load(fh)

    id2ans = {}
    for key in eval_file:
        id2ans[eval_file[key]["uuid"]] = eval_file[key]["answers"]

    for key in answer:
        print "generated:", answer[key].encode('utf-8')
        print "groundtruth:", id2ans[key]
        print
