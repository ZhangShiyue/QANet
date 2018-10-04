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
from util import get_record_parser_bidaf, convert_tokens, convert_tokens_g, evaluate, \
    evaluate_bleu, evaluate_rouge_L, evaluate_meteor, get_batch_dataset, get_dataset, \
    evaluate_rl, format_generated_questions


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
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser_bidaf(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, word_mat, char_mat, model_tpye=config.model_tpye, is_answer=config.is_answer, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

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
                loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.c: c, model.q: q if config.is_answer else a, model.a: a if config.is_answer else q,
                    model.ch: ch, model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                    model.y1: y1, model.y2: y2, model.qa_id: qa_id, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    filename = os.path.join(
                            config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)

                    metrics = evaluate_batch(config, model, config.val_num_batches,
                                             train_eval_file, sess, train_iterator, id2word,
                                             model_tpye=config.model_tpye, is_answer=config.is_answer)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/loss".format("train"), simple_value=metrics["loss"]), ])
                    writer.add_summary(loss_sum, global_step)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("train"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    em_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/em".format("train"), simple_value=metrics["exact_match"]), ])
                    writer.add_summary(em_sum, global_step)

                    metrics = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                                             dev_eval_file, sess, dev_iterator, id2word,
                                             model_tpye=config.model_tpye, is_answer=config.is_answer)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/loss".format("dev"), simple_value=metrics["loss"]), ])
                    writer.add_summary(loss_sum, global_step)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("dev"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    em_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/em".format("dev"), simple_value=metrics["exact_match"]), ])
                    writer.add_summary(em_sum, global_step)
                    writer.flush()

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


def evaluate_batch(config, model, num_batches, eval_file, sess, iterator, id2word, model_tpye="QANetModel",
                   is_answer=True, is_test=False):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        c, q, a, ch, qh, ah, y1, y2, qa_id = sess.run(next_element)
        if model_tpye == "QANetModel" or model_tpye == "TransformerModel":
            loss, byp1, byp2 = sess.run([model.loss, model.byp1, model.byp2],
                                        feed_dict={model.c: c, model.q: q, model.a: a,
                                                   model.ch: ch, model.qh: qh, model.ah: ah,
                                                   model.qa_id: qa_id, model.y1: y1, model.y2: y2})
            yp1 = map(lambda x: x[0], byp1)
            yp2 = map(lambda x: x[0], byp2)
            answer_dict_, _ = convert_tokens(eval_file, qa_id, yp1, yp2)
            answer_dict.update(answer_dict_)
        elif model_tpye == "QANetGenerator" or model_tpye == "QANetRLGenerator":
            loss, symbols = sess.run([model.loss, model.symbols],
                                     feed_dict={model.c: c, model.q: q if config.is_answer else a,
                                                model.a: a if config.is_answer else q,
                                                model.ch: ch, model.qh: qh if config.is_answer else ah,
                                                model.ah: ah if config.is_answer else qh,
                                                model.qa_id: qa_id, model.y1: y1, model.y2: y2})
            answer_dict_, _ = convert_tokens_g(eval_file, qa_id, symbols, id2word)
            answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics, f1s = evaluate(eval_file, answer_dict, is_answer=is_answer)
    metrics["loss"] = loss
    metrics["f1s"] = f1s
    if is_test:
        bleus = evaluate_bleu(eval_file, answer_dict, is_answer=is_answer)
        rougeL = evaluate_rouge_L(eval_file, answer_dict, is_answer=is_answer)
        # meteor = evaluate_meteor(eval_file, answer_dict, is_answer=is_answer)
        return metrics, bleus, rougeL
    else:
        return metrics
