import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import pickle as pkl

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")


def word_tokenize(sent, lower_word=False):
    doc = nlp(sent)
    return [token.text.lower() if lower_word else token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter=None, que_word_counter=None, char_counter=None, lower_word=False):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    max_c, max_q, max_a = 0, 0, 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                        "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                if lower_word:
                    context = context.lower()
                context_tokens = word_tokenize(context)
                max_c = max(max_c, len(context_tokens))
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                if word_counter is not None:
                    for token in context_tokens:
                        word_counter[token] += len(para["qas"])
                        for char in token:
                            char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                            "''", '" ').replace("``", '" ').replace(u'\u2013', '-')
                    if lower_word:
                        ques = ques.lower()
                    ques_tokens = ["--GO--"] + word_tokenize(ques) + ["--EOS--"]
                    max_q = max(max_q, len(ques_tokens))
                    ques_chars = [list(token) for token in ques_tokens]
                    if que_word_counter is not None:
                        for token in ques_tokens:
                            que_word_counter[token] += 1
                    elif word_counter is not None:
                        for token in ques_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    answer_tokens = []
                    answer_chars = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"].replace(u'\u2013', '-')
                        if lower_word:
                            answer_text = answer_text.lower()
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_tokens.append(["--AS--"] + word_tokenize(answer_text)+ ["--AE--"])
                        answer_chars.append([list(token) for token in answer_tokens[-1]])
                        max_a = max(max_a, len(answer_tokens[-1]))
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    context_tokens_tmp = []
                    context_chars_tmp = []
                    for i, token in enumerate(context_tokens):
                        if i == y1s[0]:
                            context_tokens_tmp.append("--AS--")
                            context_chars_tmp.append(list("--AS--"))
                        context_tokens_tmp.append(token)
                        context_chars_tmp.append(list(token))
                        if i == y2s[0]:
                            context_tokens_tmp.append("--AE--")
                            context_chars_tmp.append(list("--AE--"))
                    example = {"context_tokens_ans": context_tokens_tmp, "context_tokens": context_tokens,
                               "context_chars_ans": context_chars_tmp, "context_chars": context_chars,
                               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
                               "ans_tokens": answer_tokens, "ans_chars": answer_chars,
                               "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "context_tokens_ans": context_tokens_tmp, "context_tokens": context_tokens,
                        "spans": spans, "questions": [ques], "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        print("max_c, max_q, max_a: {}, {}, {}".format(max_c, max_q, max_a))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=0, emb_file=None, size=None, vec_size=None, size_limit=0, lower_word=False):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = counter
    if limit > 0:
        filtered_elements = {k: v for k, v in counter.items() if v > limit}
    if size_limit > 0:
        filtered_elements = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:size_limit])
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if lower_word:
                    word.lower()
                if word in filtered_elements:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
                len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                    scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
                len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    GO  = "--GO--"
    EOS = "--EOS--"
    AS  = "--AS--"
    AE = "--AE--"

    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 6)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    token2idx_dict[GO] = 2
    token2idx_dict[EOS] = 3
    token2idx_dict[AS] = 4
    token2idx_dict[AE] = 5
    embedding_dict[NULL] = np.random.normal(size=vec_size)
    embedding_dict[OOV] = np.random.normal(size=vec_size)
    embedding_dict[GO] = np.random.normal(size=vec_size)
    embedding_dict[EOS] = np.random.normal(size=vec_size)
    embedding_dict[AS] = np.random.normal(size=vec_size)
    embedding_dict[AE] = np.random.normal(size=vec_size)

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = config.test_ans_limit
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit \
               (example["y2s"][0] - example["y1s"][0]) > (ans_limit - 3)

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def build_features(config, examples, data_type, out_file, word2idx_dict, que_word2idx_dict,
                   char2idx_dict, is_test=False):
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = config.test_ans_limit if is_test else config.ans_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens_ans"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               len(example["ans_tokens"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_idxs_ans = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        context_char_idxs_ans = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_idxs_ans = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ans_idxs = np.zeros([ans_limit], dtype=np.int32)
        ans_char_idxs = np.zeros([ans_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        start, end = example["y1s"][0], example["y2s"][0]
        y1[start], y2[end] = 1.0, 1.0

        def _get_word(word, i):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_que_word(word, i):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in que_word2idx_dict:
                    return que_word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            wid = _get_word(token, i)
            if config.use_pointer:
                context_idxs[i] = len(word2idx_dict) + i if wid == 1 else wid
            else:
                context_idxs[i] = wid

        for i, token in enumerate(example["context_tokens_ans"]):
            wid = _get_word(token, i)
            if config.use_pointer:
                context_idxs_ans[i] = len(word2idx_dict) + i if wid == 1 else wid
            else:
                context_idxs_ans[i] = wid

        for i, token in enumerate(example["ques_tokens"]):
            wid = _get_que_word(token, i)
            if config.use_pointer:
                ques_idxs[i] = len(que_word2idx_dict) + example["context_tokens"].index(token) \
                    if wid == 1 and token in example["context_tokens"] else wid
                ques_idxs_ans[i] = len(que_word2idx_dict) + example["context_tokens_ans"].index(token) \
                    if wid == 1 and token in example["context_tokens_ans"] else wid
            else:
                ques_idxs[i] = wid
                ques_idxs_ans[i] = wid

        for i, token in enumerate(example["ans_tokens"][0]):
            wid = _get_word(token, i)
            if config.use_pointer:
                # there are GO and EOS in context
                ans_idxs[i] = len(word2idx_dict) + start + i if wid == 1 else wid
            else:
                ans_idxs[i] = wid

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["context_chars_ans"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs_ans[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ans_chars"][0]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ans_char_idxs[i, j] = _get_char(char)

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "context_idxs_ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs_ans.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "ques_idxs_ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs_ans.tostring()])),
            "ans_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_idxs.tostring()])),
            "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
            "context_char_idxs_ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs_ans.tostring()])),
            "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "ans_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_char_idxs.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, que_word_counter, char_counter = Counter(), Counter(), Counter()
    examples, eval = process_file(config.train_file, "train", word_counter=word_counter,
            que_word_counter=que_word_counter, char_counter=char_counter, lower_word=config.lower_word)
    train_examples, dev_examples = examples[:-11000], examples[-11000:]
    train_eval, dev_eval = {}, {}
    train_qids = []
    for example in train_examples:
        qid = str(example["id"])
        train_qids.append(qid)
        train_eval[qid] = eval[qid]
    pkl.dump(train_qids, open("train_ids.pkl", 'wb'))
    dev_qids = []
    for example in dev_examples:
        qid = str(example["id"])
        dev_qids.append(qid)
        dev_eval[qid] = eval[qid]
    pkl.dump(dev_qids, open("dev_ids.pkl", 'wb'))
    # dev_examples, dev_eval = process_file(config.dev_file, "dev", word_counter,
    #         char_counter, lower_word=config.lower_word)
    test_examples, test_eval = process_file(config.test_file, "test", lower_word=config.lower_word)

    word_emb_file = config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim,
                                                limit=config.vocab_count_limit, lower_word=config.lower_word)
    print len(word2idx_dict)
    que_word_emb_mat, que_word2idx_dict = get_embedding(que_word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim,
                                                limit=config.vocab_count_limit, lower_word=config.lower_word)
    print len(que_word2idx_dict)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, "char", emb_file=char_emb_file,
            size=char_emb_size, vec_size=char_emb_dim, limit=config.char_count_limit, lower_word=config.lower_word)
    print len(char2idx_dict)

    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict,
                   que_word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict,
                              que_word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict,
                               que_word2idx_dict, char2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.que_word_emb_file, que_word_emb_mat, message="que word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.que_word_dictionary, que_word2idx_dict, message="que word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
