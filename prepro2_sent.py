import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import nltk

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")
# nlp = spacy.load("en")

def word_tokenize(sent, lower_word=False):
    doc = nlp(sent)
    return [token.text.lower() if lower_word else token.text for token in doc]


def sent_tokenize(context):
    return [[token.text for token in nlp(sent)] for sent in nltk.sent_tokenize(context)]


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


def process_file(filename, data_type, word_counter=None, char_counter=None, lower_word=False, titles=None, total=0):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = total
    max_c, max_s, max_q, max_a = 0, 0, 0, 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            if titles is not None and article["title"].encode('utf-8') not in titles:
                continue
            for para in article["paragraphs"]:
                context = para["context"]
                if lower_word:
                    context = context.lower()
                context_tokens_sent = sent_tokenize(context)
                context_tokens = []
                for sent in context_tokens_sent:
                    context_tokens.extend(sent)
                max_c = max(max_c, len(context_tokens))
                max_s = max(max_s, max(map(len, context_tokens_sent)))
                context_chars_sent = [[list(token) for token in sent] for sent in context_tokens_sent]
                spans = convert_idx(context, context_tokens)
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"]
                    if lower_word:
                        ques = ques.lower()
                    ques_tokens = ["--GO--"] + word_tokenize(ques) + ["--EOS--"]
                    max_q = max(max_q, len(ques_tokens))
                    ques_chars = [list(token) for token in ques_tokens]
                    if word_counter is not None:
                        for token in ques_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    answer_tokens = []
                    answer_chars = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
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
                    index = 0
                    sent_start = None
                    sent_tokens = []
                    sent_tokens_ans = []
                    sent_chars = []
                    sent_chars_ans = []
                    for i, sent in enumerate(context_tokens_sent):
                        if index <= y1s[0] < index + len(sent) or index <= y2s[0] < index + len(sent):
                            if not sent_tokens:
                                sent_start = index
                            sent_tokens.extend(sent)
                            sent_chars.extend(context_chars_sent[i])
                            sent_tokens_ans.extend(sent)
                            sent_chars_ans.extend(context_chars_sent[i])
                        elif index > y2s[0]:
                            break
                        index += len(sent)
                    if word_counter is not None:
                        for token in sent_tokens:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1
                    sent_tokens_ans.insert(y1s[0] - sent_start, "--AS--")
                    sent_tokens_ans.insert(y2s[0] - sent_start + 2, "--AE--")
                    sent_chars_ans.insert(y1s[0] - sent_start, list("--AS--"))
                    sent_chars_ans.insert(y2s[0] - sent_start + 2, list("--AE--"))
                    example = {"sent_tokens_ans": sent_tokens_ans, "sent_tokens": sent_tokens,
                               "sent_chars_ans": sent_chars_ans, "sent_chars": sent_chars,
                               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
                               "ans_tokens": answer_tokens, "ans_chars": answer_chars,
                               "y1s_sent": [y1s[0] - sent_start], "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "sent_tokens_ans": sent_tokens_ans, "sent_tokens": sent_tokens,
                        "spans": spans, "questions": [ques], "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        print("max_c, max_s, max_q, max_a: {}, {}, {}, {}".format(max_c, max_s, max_q, max_a))
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
                    word = word.lower()
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

    print set(filtered_elements.keys()) - set(embedding_dict)
    exit()

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


def build_features(config, examples, data_type, out_file, word2idx_dict,
                   char2idx_dict, is_test=False):
    sent_limit = config.test_sent_limit if is_test else config.sent_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = config.test_ans_limit if is_test else config.ans_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["sent_tokens_ans"]) > sent_limit or \
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
        sent_idxs = np.zeros([sent_limit], dtype=np.int32)
        sent_idxs_ans = np.zeros([sent_limit], dtype=np.int32)
        sent_char_idxs = np.zeros([sent_limit, char_limit], dtype=np.int32)
        sent_char_idxs_ans = np.zeros([sent_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ans_idxs = np.zeros([ans_limit], dtype=np.int32)
        ans_char_idxs = np.zeros([ans_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([sent_limit], dtype=np.float32)
        y2 = np.zeros([sent_limit], dtype=np.float32)

        start_sent = example["y1s_sent"][0]

        def _get_word(word, i):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["sent_tokens"]):
            wid = _get_word(token, i)
            if config.use_pointer:
                sent_idxs[i] = len(word2idx_dict) + i if wid == 1 else wid
            else:
                sent_idxs[i] = wid

        for i, token in enumerate(example["sent_tokens_ans"]):
            wid = _get_word(token, i)
            if config.use_pointer:
                sent_idxs_ans[i] = len(word2idx_dict) + i if wid == 1 else wid
            else:
                sent_idxs_ans[i] = wid

        for i, token in enumerate(example["ques_tokens"]):
            wid = _get_word(token, i)
            if config.use_pointer:
                ques_idxs[i] = len(word2idx_dict) + example["sent_tokens_ans"].index(token) \
                    if wid == 1 and token in example["sent_tokens_ans"] else wid
            else:
                ques_idxs[i] = wid

        for i, token in enumerate(example["ans_tokens"][0]):
            wid = _get_word(token, i)
            if config.use_pointer:
                # there are GO and EOS in sentence
                ans_idxs[i] = len(word2idx_dict) + start_sent + i if wid == 1 else wid
            else:
                ans_idxs[i] = wid

        for i, token in enumerate(example["sent_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                sent_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["sent_chars_ans"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                sent_char_idxs_ans[i, j] = _get_char(char)

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
            "sent_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sent_idxs.tostring()])),
            "sent_idxs_ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sent_idxs_ans.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "ans_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ans_idxs.tostring()])),
            "sent_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sent_char_idxs.tostring()])),
            "sent_char_idxs_ans": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sent_char_idxs_ans.tostring()])),
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
    train_titles = map(lambda x: x.strip(), open("processed/doclist-train.txt", 'r').readlines())
    test_titles = map(lambda x: x.strip(), open("processed/doclist-test.txt", 'r').readlines())
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(config.train_file, "train", word_counter,
                                              char_counter, lower_word=config.lower_word, titles=train_titles)
    # dev_examples1, dev_eval1 = process_file(config.dev_file, "dev", word_counter,
    #                                         char_counter, lower_word=config.lower_word, total=len(train_examples))
    # train_examples += dev_examples1
    # train_eval.update(dev_eval1)
    dev_examples, dev_eval = process_file(config.train_file, "dev", lower_word=config.lower_word, titles=test_titles)
    test_examples, test_eval = process_file(config.train_file, "test", lower_word=config.lower_word, titles=test_titles)

    word_emb_file = config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim,
                                                limit=config.vocab_count_limit, size_limit=config.size_limit,
                                                lower_word=config.lower_word)
    print len(word2idx_dict)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, "char", emb_file=char_emb_file,
            size=char_emb_size, vec_size=char_emb_dim, limit=config.char_count_limit, lower_word=config.lower_word)
    print len(char2idx_dict)

    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict,
                               is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
