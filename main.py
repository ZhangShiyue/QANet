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
from util import get_record_parser, convert_tokens, evaluate, evaluate_bleu, get_batch_dataset, get_dataset
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
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                    model.c: c, model.q: q, model.a: a, model.ch: ch, model.qh: qh, model.ah: ah,
                    model.y1: y1, model.y2: y2,
                    model.qa_id: qa_id, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                    # loss_sum = tf.Summary(value=[tf.Summary.Value(
                    #         tag="model/gen_loss", simple_value=gen_loss), ])
                    # writer.add_summary(loss_sum, global_step)
                    # loss_sum = tf.Summary(value=[tf.Summary.Value(
                    #         tag="model/pre_loss", simple_value=pre_loss), ])
                    # writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    # _, summ = evaluate_batch(config, model, config.val_num_batches,
                    #                          train_eval_file, sess, "train", train_iterator)
                    # for s in summ:
                    #     writer.add_summary(s, global_step)
                    #
                    # metrics, summ = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                    #                                dev_eval_file, sess, "dev", dev_iterator)
                    #
                    # dev_f1 = metrics["f1"]
                    # dev_em = metrics["exact_match"]
                    # if dev_f1 < best_f1 and dev_em < best_em:
                    #     patience += 1
                    #     if patience > config.early_stop:
                    #         break
                    # else:
                    #     patience = 0
                    #     best_em = max(best_em, dev_em)
                    #     best_f1 = max(best_f1, dev_f1)
                    #
                    # for s in summ:
                    #     writer.add_summary(s, global_step)
                    # writer.flush()
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
        symbols = sess.run(model.symbols, feed_dict={model.c: c, model.q: q, model.a: a,
                                                     model.ch: ch, model.qh: qh, model.ah: ah, model.y1: y1,
                                                     model.y2: y2})
        symbols = list(symbols)
        if 3 in symbols:
            symbols = symbols[:symbols.index(3)]
        answer = u' '.join([id2word[symbol] for symbol in symbols[1:]])
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
        answer_dict_ = {str(qa_id[0]): answer}
        answer_dict.update(answer_dict_)
    metrics = evaluate(eval_file, answer_dict)
    f1_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
            tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [f1_sum, em_sum]

    # answer_dict = {}
    # losses = []
    # next_element = iterator.get_next()
    # for _ in tqdm(range(1, num_batches + 1)):
    #     c, q, a, ch, qh, ah, y1, y2, qa_id = sess.run(next_element)
    #     qa_id, loss, yp1, yp2, = sess.run([model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={model.c: c, model.q: q, model.a: a,
    #                                                                 model.ch: ch, model.qh: qh, model.ah: ah,
    #                                                                 model.qa_id: qa_id, model.y1: y1, model.y2: y2})
    #     answer_dict_, _ = convert_tokens(
    #             eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
    #     answer_dict.update(answer_dict_)
    #     losses.append(loss)
    # loss = np.mean(losses)
    # metrics = evaluate(eval_file, answer_dict)
    # metrics["loss"] = loss
    # loss_sum = tf.Summary(value=[tf.Summary.Value(
    #         tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    # f1_sum = tf.Summary(value=[tf.Summary.Value(
    #         tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    # em_sum = tf.Summary(value=[tf.Summary.Value(
    #         tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    # return metrics, [loss_sum, f1_sum, em_sum]


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
                yp1, yp2 = sess.run([model.yp2, model.yp2], feed_dict={model.c: c, model.q: q, model.a: a,
                                                             model.ch: ch, model.qh: qh, model.ah: ah, model.y1: y1,
                                                             model.y2: y2, model.qa_id: qa_id})
                # context = eval_file[str(qa_id[0])]["context"].replace(
                #         "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                # context_tokens = word_tokenize(context)
                # bsymbols = zip(*bsymbols)
                answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                # answers = []
                # for symbols, prev_prob in zip(bsymbols, prev_probs):
                #     symbols = list(symbols)
                #     if 3 in symbols:
                #         symbols = symbols[:symbols.index(3)]
                #     answer = u' '.join([id2word[symbol] if symbol in id2word
                #                         else context_tokens[symbol - len(id2word)] for symbol in symbols])
                #     # deal with special symbols like %, $ etc
                #     elim_pre_spas = [u' %', u" 's", u' ,']
                #     for s in elim_pre_spas:
                #         if s in answer:
                #             answer = s[1:].join(answer.split(s))
                #     elim_beh_spas = [u'$ ', u'\xa3 ', u'# ']
                #     for s in elim_beh_spas:
                #         if s in answer:
                #             answer = s[:-1].join(answer.split(s))
                #     elim_both_spas = [u' - ']
                #     for s in elim_both_spas:
                #         if s in answer:
                #             answer = s[1:-1].join(answer.split(s))
                #     answers.append(answer)
                # answer_dict_ = {str(qa_id[0]): answers[0]}
                # answer_dict.update(answer_dict_)
            metrics = evaluate(eval_file, answer_dict, is_answer=True)
            with open("{}_b{}.json".format(config.answer_file, config.beam_size), "w") as fh:
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
    with open(config.rerank_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.rerank_file, get_record_parser(
                config, len(word_mat) + config.test_para_limit, is_test=True, is_rerank=True),
                                 config, is_test=True).make_one_shot_iterator()
        test_next_element = test_batch.get_next()
        model = Model(config, word_mat, char_mat, trainable=True, rerank=True, graph=g)

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
                c, q, a, ch, qh, ah, y1, y2, qa_id, can_id = sess.run(test_next_element)
                batch_loss = sess.run(model.batch_loss, feed_dict={model.c: c, model.q: q, model.a: a, model.ch: ch,
                                                       model.qh: qh, model.ah: ah, model.y1: y1, model.y2: y2,
                                                       model.qa_id: qa_id, model.can_id: can_id})
                for qid, cid, l in zip(qa_id, can_id, batch_loss):
                    scores[(str(qid), str(cid))] = str(l)
            with open(config.listener_score_file, "w") as fh:
                json.dump(scores, fh)


def test_bleu(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open("{}_b{}.json".format(config.question_file, config.beam_size), "r") as fh:
        answer = json.load(fh)

    groundtruths, answers = evaluate_bleu(eval_file, answer)
    with open("{}_b{}_generated".format(config.question_file, config.beam_size), 'w') as f:
        f.write('\n'.join([' '.join(answer) for answer in answers]).encode('utf-8'))
    with open("{}_b{}_groundtruth".format(config.question_file, config.beam_size), 'w') as f:
        f.write('\n'.join([' '.join(answer) for answer in groundtruths]).encode('utf-8'))


def test_reranked(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.listener_score_file, "r") as fh:
        listener_score_file = json.load(fh)
    with open(config.beam_search_file, "r") as fh:
        beam_search_file = json.load(fh)

    answer_dict = {}
    for qid in beam_search_file:
        if qid not in answer_dict:
            answer_dict[qid] = []
        for cid, (ans, _, _, prob) in enumerate(beam_search_file[qid]):
            score = float(prob) + 1.3 * float(listener_score_file[str((str(int(qid)), str(cid)))])
            answer_dict[qid].append((ans, score))
    answer_dict = {qid: sorted(answer_dict[qid], key=lambda x: x[1])[0][0] for qid in answer_dict}

    metrics = evaluate(eval_file, answer_dict, is_answer=True)
    print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))


def tmp(config):
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open("{}_b{}.json".format(config.answer_file, config.beam_size), "r") as fh:
        answer = json.load(fh)

    for qid in answer:
        if len(word_tokenize(answer[qid])) > 1:
            print answer[qid].encode("utf-8")
    # metrics = evaluate(eval_file, answer, is_answer=True)
    # print("D: Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))

    # for qid in beam_search_file:
    #     print qid
    #     for ans, _, _, _ in beam_search_file[qid]:
    #         if ans in eval_file[qid]["answers"]:
    #             print ans.encode("utf-8"), "*"
    #         else:
    #             print ans.encode("utf-8")
    #     print