"""
extract questions
"""
import json
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from prepro import word_tokenize, save


def squad(filename):
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in source["data"]:
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    ques = qa["question"].replace(
                            "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                    print ques.encode("utf-8")


def quora(filename):
    for line in open(filename):
        items = line.strip().split('\t')
        if items[0] != 'id':
            try:
                print items[3]
                print items[4]
            except:
                continue


def ms_marco(filename):
    with open(filename, "r") as fh:
        source = json.load(fh)
        query = source[u'query']
        for id in query:
            print query[id].encode("utf-8")


def build_features(config, is_test=False):
    with open(config.word_dictionary, "r") as fh:
        word2idx_dict = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char2idx_dict = json.load(fh)
    ques_limit = config.ques_limit
    char_limit = config.char_limit
    writer = tf.python_io.TFRecordWriter(config.question_train_record_file if not is_test else
                                         config.question_dev_record_file)

    def _get_word(word, i):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    total_ = 0
    examples = []
    for filename in (config.question_train_files if not is_test else config.question_dev_files):
        for line in open(filename):
            total_ += 1
            question = line.strip().decode("utf-8")
            if config.lower_word:
                question = question.lower()
            ques_tokens = ["--GO--"] + word_tokenize(question) + ["--EOS--"]
            if len(ques_tokens) > ques_limit:
                continue
            ques_chars = [list(token) for token in ques_tokens]
            example = {"ques_tokens": ques_tokens, "ques_chars": ques_chars}
            examples.append(example)
    random.shuffle(examples)

    total = 0
    meta = {}
    for example in tqdm(examples):
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

        unk_count = 0
        for i, token in enumerate(example["ques_tokens"]):
            wid = _get_word(token, i)
            if wid == 1: unk_count += 1
            ques_idxs[i] = wid

        if (unk_count + 0.0) / len(example["ques_tokens"]) > 0.2: continue

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        total += 1
        record = tf.train.Example(features=tf.train.Features(feature={
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[total]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def prepro(config):
    build_features(config, is_test=False)
    # dev_meta = build_features(config, is_test=True)
    # save(config.question_dev_meta, dev_meta, message="dev meta")


if __name__ == '__main__':
    squad("/playpen1/home/shiyue/QANet/squad/dev-v1.1.json")
    # quora("/playpen1/home/shiyue/QANet/squad/QQP/original/quora_duplicate_questions.tsv")
    # ms_marco("/playpen1/home/shiyue/QANet/squad/MS MACRO/dev_v2.1.json")
