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
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 100.0
        patience = 0
        best_f1 = 0.
        best_em = 0.
        sloss = 0.0

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                    handle: train_handle, model.dropout: config.dropout})
                sloss += loss / config.checkpoint
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    print("sloss: {}".format(sloss))
                    sloss = 0.0
                    _, summ = evaluate_batch(
                            model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(
                            model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

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


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo=True)
    demo = Demo(model, config)


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
        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict_g = {}
            answer_dict_d = {}
            res_g_b = {}
            res_d_b = {}
            for step in tqdm(range(total // config.test_batch_size + 1)):
                qa_id, loss, yp1, yp2, byp1, byp2, symbols = sess.run(
                        [model.qa_id, model.loss, model.yp1, model.yp2, model.byp1, model.byp2, model.symbols])
                bsymbols = zip(*symbols)
                answers = []
                for symbols in bsymbols:
                    if 2 in symbols:
                        symbols = symbols[:symbols.index(2)]
                    context = eval_file[str(qa_id[0])]["context"].replace(
                            "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                    context_tokens = word_tokenize(context)
                    answer = u' '.join([id2word[symbol] if symbol in id2word
                                        else context_tokens[symbol - len(id2word)] for symbol in symbols])
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
                    answers.append(answer)
                res_g_b[str(qa_id[0])] = answers
                answer_dict_g_ = {str(qa_id[0]): answers[0]}
                # remapped_dict_ = {eval_file[str(qa_id[0])]["uuid"]: answers[0]}
                answer_dict_g.update(answer_dict_g_)

                # answer_dict_d_, _ = convert_tokens(
                #     eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                # remapped_dict.update(remapped_dict_)
                losses.append(loss)
                # ==== get prediction model beam search results ====
                answers = []
                for yp1, yp2 in zip(byp1, byp2):
                    answer_dict_, remapped_dict_ = convert_tokens(
                        eval_file, qa_id.tolist(), [yp1], [yp2])
                    answers.append(answer_dict_.values()[0])
                res_d_b[str(qa_id[0])] = answers
                answer_dict_d_ = {str(qa_id[0]): answers[0]}
                answer_dict_d.update(answer_dict_d_)

            save("{}{}.json".format(config.res_g_b_file, config.beam_size), res_g_b, "res_g_b")
            save("{}{}.json".format(config.res_d_b_file, config.beam_size), res_d_b, "res_d_b")

            loss = np.mean(losses)
            metrics = evaluate(eval_file, answer_dict_d)
            with open("{}_d_b{}.json".format(config.answer_file, config.beam_size), "w") as fh:
                json.dump(answer_dict_d, fh)
            print("D: Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))
            metrics = evaluate(eval_file, answer_dict_g)
            with open("{}_g_b{}.json".format(config.answer_file, config.beam_size), "w") as fh:
                json.dump(answer_dict_g, fh)
            print("G: Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))



def test_beam(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.res_g_b_file, "r") as fh:
        g_answer_dict = json.load(fh)
    with open(config.res_d_b_file, "r") as fh:
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
    with open(config.res_d_b_file, "r") as fh:
        d_answer_dict = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.rerank_file, get_record_parser(
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
            with open(config.answer_file, "w") as fh:
                json.dump(answer_dict, fh)
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
