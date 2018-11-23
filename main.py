import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from model import Model
from util import get_record_parser, convert_tokens, convert_tokens_g, evaluate, \
    evaluate_bleu, evaluate_rouge_L, evaluate_meteor, get_batch_dataset, get_dataset, \
    evaluate_rl, evaluate_rl_dual, format_generated_questions, format_sampled_questions


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
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, word_mat, char_mat, model_tpye=config.model_tpye, is_answer=config.is_answer, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

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
                c, ca, cau, q, qa, a, ch, cha, qh, ah, y1, y2, qa_id = sess.run(train_next_element)
                loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.c: c if config.is_answer else ca, model.cu: cau,
                    model.q: q if config.is_answer else a,
                    model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                    model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
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
                    em_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/em".format("train"), simple_value=metrics["exact_match"]), ])
                    writer.add_summary(em_sum, global_step)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("train"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    if not config.is_answer:
                        bleu_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/bleu".format("train"), simple_value=metrics["bleu"][0]*100), ])
                        writer.add_summary(bleu_sum, global_step)
                        rougeL_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/rougeL".format("train"), simple_value=metrics["rougeL"]*100), ])
                        writer.add_summary(rougeL_sum, global_step)
                        meteor_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/meteor".format("train"), simple_value=metrics["meteor"][0]*100), ])
                        writer.add_summary(meteor_sum, global_step)

                    metrics = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                                             dev_eval_file, sess, dev_iterator, id2word,
                                             model_tpye=config.model_tpye, is_answer=config.is_answer)
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/loss".format("dev"), simple_value=metrics["loss"]), ])
                    writer.add_summary(loss_sum, global_step)
                    em_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/em".format("dev"), simple_value=metrics["exact_match"]), ])
                    writer.add_summary(em_sum, global_step)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("dev"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    if not config.is_answer:
                        bleu_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/bleu".format("dev"), simple_value=metrics["bleu"][0]*100), ])
                        writer.add_summary(bleu_sum, global_step)
                        rougeL_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/rougeL".format("dev"), simple_value=metrics["rougeL"]*100), ])
                        writer.add_summary(rougeL_sum, global_step)
                        meteor_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/meteor".format("dev"), simple_value=metrics["meteor"][0]*100), ])
                        writer.add_summary(meteor_sum, global_step)
                    writer.flush()


def train_rl(config):
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
    baseline_file = None
    if config.if_fix_base:
        with open(config.baseline_file, "r") as fh:
            baseline_file = json.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
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
            start_step = global_step
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                c, ca, q, qa, a, ch, cha, qh, ah, y1, y2, qa_id = sess.run(train_next_element)
                if global_step < config.pre_step:
                    loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                        model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                        model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                        model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                        model.y1: y1, model.y2: y2, model.qa_id: qa_id, model.dropout: config.dropout})
                else:
                    symbols, symbols_rl = sess.run([model.symbols, model.symbols_rl], feed_dict={
                        model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                        model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                        model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                        model.y1: y1, model.y2: y2, model.qa_id: qa_id})
                    reward, reward_rl, reward_base = evaluate_rl(train_eval_file, qa_id, symbols, symbols_rl, id2word,
                                                                 baseline_file=baseline_file, is_answer=config.is_answer,
                                                                 metric=config.rl_metric, if_fix_base=config.if_fix_base)
                    sa = format_sampled_questions(symbols_rl, config.batch_size, config.ques_limit)
                    loss_ml, loss_rl, _ = sess.run([model.loss_ml, model.loss_rl, model.train_op], feed_dict={
                        model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                        model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                        model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                        model.y1: y1, model.y2: y2, model.qa_id: qa_id, model.dropout: config.dropout,
                        model.sa: sa, model.reward: reward})
                if global_step == start_step + 1 or global_step % config.period == 0:
                    loss_ml_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss_ml", simple_value=loss_ml), ])
                    writer.add_summary(loss_ml_sum, global_step)
                    reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/reward_rl", simple_value=reward_rl), ])
                    writer.add_summary(reward_rl_sum, global_step)
                    reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/reward_base", simple_value=reward_base), ])
                    writer.add_summary(reward_base_sum, global_step)
                    writer.flush()
                if global_step % config.checkpoint == 0:
                    filename = os.path.join(
                            config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)
                    metrics = evaluate_batch(config, model, config.val_num_batches,
                                             train_eval_file, sess, train_iterator, id2word,
                                             model_tpye=config.model_tpye, is_answer=config.is_answer)
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("train"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    bleu_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/bleu".format("train"), simple_value=metrics["bleu"][0]*100), ])
                    writer.add_summary(bleu_sum, global_step)
                    rougeL_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/rougeL".format("train"), simple_value=metrics["rougeL"]*100), ])
                    writer.add_summary(rougeL_sum, global_step)
                    meteor_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/meteor".format("train"), simple_value=metrics["meteor"][0]*100), ])
                    writer.add_summary(meteor_sum, global_step)

                    metrics = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                                                   dev_eval_file, sess, dev_iterator, id2word,
                                             model_tpye=config.model_tpye, is_answer=config.is_answer)
                    # print metrics["f1"], metrics["bleu"][0], metrics["rougeL"], metrics["meteor"][0]
                    # exit()
                    f1_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/f1".format("dev"), simple_value=metrics["f1"]), ])
                    writer.add_summary(f1_sum, global_step)
                    bleu_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/bleu".format("dev"), simple_value=metrics["bleu"][0]*100), ])
                    writer.add_summary(bleu_sum, global_step)
                    rougeL_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/rougeL".format("dev"), simple_value=metrics["rougeL"]*100), ])
                    writer.add_summary(rougeL_sum, global_step)
                    meteor_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/meteor".format("dev"), simple_value=metrics["meteor"][0]*100), ])
                    writer.add_summary(meteor_sum, global_step)
                    writer.flush()

                    dev_f1 = metrics["f1"]
                    if dev_f1 < best_f1:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_f1 = max(best_f1, dev_f1)


def train_dual(config):
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
    with open(config.char_dictionary, "r") as fh:
        char2idx_dict = json.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    graph_dual = tf.Graph()
    with graph.as_default():
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, word_mat, char_mat, model_tpye=config.model_tpye, is_answer=config.is_answer, graph=graph)
    dual_model = Model(config, word_mat, char_mat, model_tpye=config.dual_model_tpye, is_answer=config.is_answer_dual, graph=graph_dual)

    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth = True

    patience = 0
    best_f1 = 0.
    best_em = 0.

    sess = tf.Session(graph=graph)
    sess_dual = tf.Session(graph=graph_dual)

    writer = tf.summary.FileWriter(config.log_dir)
    with sess.as_default():
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
    with sess_dual.as_default():
        with graph_dual.as_default():
            sess_dual.run(tf.global_variables_initializer())
            saver_dual = tf.train.Saver(max_to_keep=1000)
            if os.path.exists(os.path.join(config.save_dir_dual, "checkpoint")):
                saver_dual.restore(sess_dual, tf.train.latest_checkpoint(config.save_dir_dual))
    global_step = max(sess.run(model.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess.run(model.global_step) + 1
        c, ca, q, qa, a, ch, cha, qh, ah, y1, y2, qa_id = sess.run(train_next_element)
        if global_step < config.pre_step:
            loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                model.y1: y1, model.y2: y2, model.qa_id: qa_id, model.dropout: config.dropout})
        else:
            # samples for reward computing
            symbols, symbols_rl = sess.run([model.symbols, model.symbols_rl], feed_dict={
                model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                model.y1: y1, model.y2: y2, model.qa_id: qa_id})
            # format sample for QA
            ques_idxs, ques_char_idxs, ques_idxs_rl, ques_char_idxs_rl = \
                format_generated_questions(train_eval_file, qa_id, symbols, symbols_rl, config.batch_size,
                                           config.ques_limit, config.char_limit, id2word, char2idx_dict)
            # QA reward
            base_dual_loss, base_dual_byp1, base_dual_byp2 = sess_dual.run(
                    [dual_model.batch_loss, dual_model.byp1, dual_model.byp2], feed_dict={
                    dual_model.c: c, dual_model.q: ques_idxs, dual_model.a: a,
                    dual_model.ch: ch, dual_model.qh: ques_char_idxs, dual_model.ah: ah,
                    dual_model.y1: y1, dual_model.y2: y2, dual_model.qa_id: qa_id})
            dual_loss, dual_byp1, dual_byp2 = sess_dual.run(
                    [dual_model.batch_loss, dual_model.byp1, dual_model.byp2], feed_dict={
                    dual_model.c: c, dual_model.q: ques_idxs_rl, dual_model.a: a,
                    dual_model.ch: ch, dual_model.qh: ques_char_idxs_rl, dual_model.ah: ah,
                    dual_model.y1: y1, dual_model.y2: y2, dual_model.qa_id: qa_id})
            reward, reward_rl, reward_base = evaluate_rl_dual(train_eval_file, qa_id, base_dual_byp1,
                                                              base_dual_byp2, dual_byp1, dual_byp2,
                                                              base_dual_loss, dual_loss,
                                                              config.dual_rl_metric, config.has_baseline)
            # train with rl
            sa = format_sampled_questions(symbols_rl, config.batch_size, config.ques_limit)
            loss_ml, _ = sess.run([model.loss_ml, model.train_op], feed_dict={
                model.c: c if config.is_answer else ca, model.q: q if config.is_answer else a,
                model.a: a if config.is_answer else qa, model.ch: ch if config.is_answer else cha,
                model.qh: qh if config.is_answer else ah, model.ah: ah if config.is_answer else qh,
                model.y1: y1, model.y2: y2, model.qa_id: qa_id, model.dropout: config.dropout,
                model.sa: sa, model.reward: reward})
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss_ml", simple_value=loss_ml), ])
            writer.add_summary(loss_sum, global_step)
            reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_base", simple_value=reward_base), ])
            writer.add_summary(reward_base_sum, global_step)
            reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_rl", simple_value=reward_rl), ])
            writer.add_summary(reward_rl_sum, global_step)
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


def evaluate_batch(config, model, num_batches, eval_file, sess, iterator, id2word, model_tpye="QANetModel",
                   is_answer=True, is_test=False):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        c, ca, cau, q, qa, a, ch, cha, qh, ah, y1, y2, qa_id = sess.run(next_element)
        if model_tpye == "QANetModel" or model_tpye == "BiDAFModel":
            loss, byp1, byp2 = sess.run([model.loss, model.byp1, model.byp2],
                                        feed_dict={model.c: c, model.q: q, model.a: a,
                                                   model.ch: ch, model.qh: qh, model.ah: ah,
                                                   model.qa_id: qa_id, model.y1: y1, model.y2: y2})
            yp1 = map(lambda x: x[0], byp1)
            yp2 = map(lambda x: x[0], byp2)
            answer_dict_, _ = convert_tokens(eval_file, qa_id, yp1, yp2)
            answer_dict.update(answer_dict_)
        elif model_tpye == "QANetGenerator" or model_tpye == "QANetRLGenerator" \
                or model_tpye == "BiDAFGenerator" or model_tpye == "BiDAFRLGenerator":
            loss, symbols = sess.run([model.loss, model.symbols],
                                     feed_dict={model.c: c if config.is_answer else ca,
                                                model.cu: cau,
                                                model.q: q if config.is_answer else a,
                                                model.a: a if config.is_answer else qa,
                                                model.ch: ch if config.is_answer else cha,
                                                model.qh: qh if config.is_answer else ah,
                                                model.ah: ah if config.is_answer else qh,
                                                model.qa_id: qa_id, model.y1: y1, model.y2: y2})
            answer_dict_, _ = convert_tokens_g(eval_file, qa_id, symbols, id2word)
            answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics, f1s = evaluate(eval_file, answer_dict, is_answer=is_answer)
    metrics["loss"] = loss
    metrics["f1s"] = f1s
    if not is_answer:
        bleu = evaluate_bleu(eval_file, answer_dict, is_answer=is_answer)
        rougeL = evaluate_rouge_L(eval_file, answer_dict, is_answer=is_answer)
        meteor = evaluate_meteor(eval_file, answer_dict, is_answer=is_answer)
        metrics["bleu"] = bleu
        metrics["rougeL"] = rougeL
        metrics["meteor"] = meteor
    return metrics


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
        test_iterator = get_dataset(config.test_record_file, get_record_parser(
                config, is_test=True), config, is_test=True).make_one_shot_iterator()
        model = Model(config, word_mat, char_mat, model_tpye=config.model_tpye, trainable=False,
                      is_answer=config.is_answer, graph=g)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter("{}/beam{}".format(config.log_dir, config.beam_size))
            for ckpt in range(4, config.num_steps / config.checkpoint + 1):
                checkpoint = "{}/model_{}.ckpt".format(config.save_dir, ckpt*config.checkpoint)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint)
                # if config.decay < 1.0:
                #     sess.run(model.assign_vars)
                global_step = sess.run(model.global_step)
                metrics = evaluate_batch(config, model, total // config.test_batch_size + 1,
                                         eval_file, sess, test_iterator, id2word, model_tpye=config.model_tpye,
                                         is_answer=config.is_answer, is_test=True)

                loss_sum = tf.Summary(value=[tf.Summary.Value(
                                tag="{}/loss".format("test"), simple_value=metrics["loss"]), ])
                writer.add_summary(loss_sum, global_step)
                f1_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="{}/f1".format("test"), simple_value=metrics["f1"]), ])
                writer.add_summary(f1_sum, global_step)
                em_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="{}/em".format("test"), simple_value=metrics["exact_match"]), ])
                writer.add_summary(em_sum, global_step)
                if not config.is_answer:
                    bleu_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/bleu".format("test"), simple_value=metrics["bleu"][0]*100), ])
                    writer.add_summary(bleu_sum, global_step)
                    rougeL_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/rougeL".format("test"), simple_value=metrics["rougeL"]*100), ])
                    writer.add_summary(rougeL_sum, global_step)
                    meteor_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="{}/meteor".format("test"), simple_value=metrics["meteor"][0]*100), ])
                    writer.add_summary(meteor_sum, global_step)
                writer.flush()


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
    with open("{}.json".format(config.answer_file), "r") as fh:
        answer = json.load(fh)

    groundtruths, answers = evaluate_bleu(eval_file, answer)
    with open("{}_b{}_generated".format(config.answer_file, config.beam_size), 'w') as f:
        f.write('\n'.join([' '.join(answer) for answer in answers]).encode('utf-8'))
    with open("{}_b{}_groundtruth".format(config.answer_file, config.beam_size), 'w') as f:
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
    from util import normalize_answer, word_tokenize
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)
    print len(test_eval_file)
    questions = []
    for key in test_eval_file:
        que = test_eval_file[key]["questions"][0]
        questions.append(' '.join(word_tokenize(que.strip().lower())))
    questions = sorted(questions)

    lines = open("processed/tgt-test.txt", 'r').readlines()
    lines = sorted(map(lambda x: x.strip().lower(), lines))

    for que, line in zip(questions, lines):
        print que
        print line
        print
        exit()