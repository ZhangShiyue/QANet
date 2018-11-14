import os
import tensorflow as tf
from prepro import prepro
from prepro1 import prepro as prepro1
from prepro2 import prepro as prepro2
from prepro3 import prepro as prepro3
# from prepro2_sent import prepro as prepro2_sent
from main import train, train_rl, train_dual, test, test_beam, test_bleu, \
    test_rerank, test_reranked, tmp

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

flags = tf.flags

# home = os.path.expanduser("/nlp/shiyue/QANet/")
home = os.path.expanduser("/playpen1/home/shiyue/QANet/")
train_file = os.path.join(home, "squad", "train-v1.1.json")
dev_file = os.path.join(home, "squad", "dev-v1.1.json")
test_file = os.path.join(home, "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "glove", "glove.840B.300d.txt")

train_dir = "train"
model_name = "BiDAF"
dir_name = os.path.join(train_dir, model_name)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
    os.mkdir(os.path.join(os.getcwd(),dir_name))
target_dir = "data_new_sent"
log_dir = os.path.join(dir_name, "event_qg_sent14")
save_dir = os.path.join(dir_name, "model_qg_sent14")
save_dir_dual = os.path.join(dir_name, "model_qa")
answer_dir = os.path.join(dir_name, "answer_qg_sent14")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
que_word_emb_file = os.path.join(target_dir, "que_word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
que_word_dictionary = os.path.join(target_dir, "que_word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
answer_file = os.path.join(answer_dir, "answer")
baseline_file = os.path.join(dir_name, "sanswer_que_gen/baseline_f1.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("model_tpye", "BiDAFGenerator", "Model type")
flags.DEFINE_string("dual_model_tpye", "BiDAFModel", "Model type")
flags.DEFINE_string("attention_tpye", "location", "Model type")
flags.DEFINE_boolean("is_sent", True, "Input sentence or paragraph")
flags.DEFINE_boolean("is_answer", False, "Output answer or question")
flags.DEFINE_boolean("is_answer_dual", True, "Output answer or question")
flags.DEFINE_string("rl_metric", "meteor", "The metric used to train rl")
flags.DEFINE_string("dual_rl_metric", "f1", "The metric used to train rl")
flags.DEFINE_string("baseline_type", "beam", "The sampling strategy used when producing baseline")
flags.DEFINE_boolean("has_baseline", True, "Use baseline or not")
flags.DEFINE_boolean("if_fix_base", False, "Fix baseline or not")
flags.DEFINE_boolean("word_trainable", False, "Train word embeddings along or not")
flags.DEFINE_boolean("use_pointer", True, "Use pointer network or not")

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("save_dir_dual", save_dir_dual, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("que_word_emb_file", que_word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")
flags.DEFINE_string("baseline_file", baseline_file, "baseline f1 scores")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("que_word_dictionary", que_word_dictionary, "Word dictionary")
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("test_batch_size", 32, "Batch size")
flags.DEFINE_integer("beam_size", 1, "Beam size")
flags.DEFINE_integer("num_steps", 30000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 1000, "period to save batch loss")
flags.DEFINE_integer("pre_step", 30000, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 32, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.3, "Dropout prob across the layers")
flags.DEFINE_float("mixing_ratio", 0.9, "The mixing ratio between ml loss and rl loss")
flags.DEFINE_float("answer_sup_ratio", None, "The mixing ratio between ml loss and rl loss")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("ml_learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("rl_learning_rate", 0.00001, "Learning rate")
flags.DEFINE_float("decay", None, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-07, "L2 norm scale")
flags.DEFINE_integer("hidden", 600, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")

flags.DEFINE_integer("model_encoder_layers", 3, "The number of model encoder")
flags.DEFINE_integer("model_encoder_blocks", 2, "The number of model encoder")
flags.DEFINE_integer("model_encoder_convs", 2, "The number of model encoder")
flags.DEFINE_integer("input_encoder_convs", 2, "The number of model encoder")
flags.DEFINE_integer("decoder_layers", 2, "The number of model decoder")

# preprocess data
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")
flags.DEFINE_boolean("lower_word", False, "Whether to lower word")
flags.DEFINE_integer("vocab_count_limit", -1, "Minimum count of words in the vocab")
flags.DEFINE_integer("char_count_limit", -1, "Minimum count of chars in the char vocab")
flags.DEFINE_integer("size_limit", -1, "Minimum count of chars in the char vocab")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("sent_limit", 100, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_sent_limit", 200, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("test_ans_limit", 50, "Limit length for answer in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "train_rl":
        train_rl(config)
    elif config.mode == "train_dual":
        train_dual(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "prepro1":
        prepro1(config)
    elif config.mode == "prepro2":
        prepro2(config)
    elif config.mode == "prepro2_sent":
        prepro2_sent(config)
    elif config.mode == "prepro3":
        prepro3(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "test_beam":
        test_beam(config)
    elif config.mode == "test_bleu":
        test_bleu(config)
    elif config.mode == "test_rerank":
        test_rerank(config)
    elif config.mode == "test_reranked":
        test_reranked(config)
    elif config.mode == "tmp":
        tmp(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
