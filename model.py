import tensorflow as tf
from layers import regularizer, initializer, total_params
from qanet import QANetModel, QANetGenerator, QANetRLGenerator
from bidaf import BiDAFModel, BiDAFGenerator, BiDAFRLGenerator


class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, model_tpye="QANetModel", trainable=True, is_answer=True, graph=None):

        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.N = config.batch_size if trainable else config.test_batch_size
            self.PL = config.para_limit if trainable else config.test_para_limit
            self.QL = config.ques_limit if trainable else config.test_ques_limit
            self.AL = config.ans_limit if trainable else config.test_ans_limit
            self.CL = config.char_limit
            if not is_answer:
                self.QL, self.AL = self.AL, self.QL

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.qa_id = tf.placeholder(tf.int32, [self.N], "qa_id")
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.c = tf.placeholder(tf.int32, [self.N, self.PL], "context")
            self.q = tf.placeholder(tf.int32, [self.N, self.QL], "question")
            self.a = tf.placeholder(tf.int32, [self.N, self.AL], "answer")
            self.ch = tf.placeholder(tf.int32, [self.N, self.PL, self.CL], "context_char")
            self.qh = tf.placeholder(tf.int32, [self.N, self.QL, self.CL], "question_char")
            self.ah = tf.placeholder(tf.int32, [self.N, self.AL, self.CL], "answer_char")
            self.y1 = tf.placeholder(tf.int32, [self.N, self.PL], "answer_index1")
            self.y2 = tf.placeholder(tf.int32, [self.N, self.PL], "answer_index2")
            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.a_mask = tf.cast(self.a, tf.bool)

            self.reward = tf.placeholder_with_default(tf.ones([self.N]), (self.N), name="reward")
            self.sa = tf.placeholder_with_default(tf.zeros([self.N, self.AL], dtype=tf.int32),
                                                  (self.N, self.AL), name="sampled_answer")

            # self.word_unk = tf.get_variable("word_unk", shape=[1, config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=config.word_trainable)
            # oov = tf.stop_gradient(tf.nn.embedding_lookup(original_word_mat, [1]))
            # additional_word_mat = tf.tile(oov, [self.PL, 1])
            # self.word_mat = tf.concat([original_word_mat, additional_word_mat], axis=0)

            self.num_words = len(word_mat)
            self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            if model_tpye == "QANetModel":
                model = QANetModel(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh, self.y1, self.y2,
                                   self.word_mat, self.char_mat, self.dropout, self.N, self.PL, self.QL, self.AL,
                                   self.CL, config.hidden, config.char_dim, config.glove_dim, config.num_heads,
                                   config.model_encoder_layers, config.model_encoder_blocks,
                                   config.model_encoder_convs, config.input_encoder_convs)
                self.loss, self.batch_loss = model.build_model(self.global_step)
                self.byp1, self.byp2, self.bprobs = model.sample(config.beam_size)
                self.lr = tf.minimum(config.ml_learning_rate, 0.001 / tf.log(999.) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
            elif model_tpye == "BiDAFModel":
                model = BiDAFModel(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh, self.y1, self.y2,
                                   self.word_mat, self.char_mat, self.dropout, self.N, self.PL, self.QL, self.AL,
                                   self.CL, config.hidden, config.char_dim, config.glove_dim)
                self.loss, self.batch_loss = model.build_model(self.global_step)
                self.byp1, self.byp2, self.bprobs = model.sample(config.beam_size)
                self.lr = tf.minimum(config.ml_learning_rate, 0.001 / tf.log(999.) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
            elif model_tpye == "QANetGenerator":
                model = QANetGenerator(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh,
                                       self.a, self.a_mask, self.ah, self.y1, self.y2, self.word_mat,
                                       self.char_mat, self.num_words, self.dropout, self.N, self.PL, self.QL,
                                       self.AL, self.CL, config.hidden, config.char_dim,
                                       config.glove_dim, config.num_heads, config.model_encoder_layers,
                                       config.model_encoder_blocks, config.model_encoder_convs,
                                       config.input_encoder_convs, config.use_pointer)
                self.loss = model.build_model(self.global_step)
                self.symbols = model.sample(config.beam_size)
                self.lr = tf.minimum(config.ml_learning_rate, 0.001 / tf.log(999.) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
            elif model_tpye == "BiDAFGenerator":
                model = BiDAFGenerator(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh, self.a, self.a_mask,
                                       self.ah, self.y1, self.y2, self.word_mat, self.char_mat, self.dropout,
                                       self.N, self.PL, self.QL, self.AL, self.CL, config.hidden, config.char_dim,
                                       config.glove_dim, self.num_words, config.use_pointer, config.attention_tpye)
                self.loss, self.batch_loss = model.build_model(self.global_step)
                self.symbols = model.sample(config.beam_size)
                self.lr = tf.minimum(config.ml_learning_rate, 0.001 / tf.log(999.) *
                                     tf.log(tf.cast(self.global_step, tf.float32) + 1))
            elif model_tpye == "QANetRLGenerator":
                model = QANetRLGenerator(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh,
                                         self.a, self.a_mask, self.ah, self.y1, self.y2, self.word_mat,
                                         self.char_mat, self.num_words, self.dropout, self.N, self.PL, self.QL,
                                         self.AL, self.CL, config.hidden, config.char_dim, config.glove_dim,
                                         config.num_heads, config.model_encoder_layers, config.model_encoder_blocks,
                                         config.model_encoder_convs, config.input_encoder_convs, self.reward,
                                         self.sa, config.mixing_ratio, config.pre_step)
                self.loss, self.loss_ml, self.loss_rl = model.build_model(self.global_step)
                self.symbols = model.sample(config.beam_size) if config.baseline_type == "beam" else model.sample_rl()
                self.symbols_rl = model.sample_rl()
                self.lr = tf.cond(self.global_step < config.pre_step, lambda: tf.minimum(config.ml_learning_rate,
                                  0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1)),
                                  lambda: config.rl_learning_rate)
            elif model_tpye == "BiDAFRLGenerator":
                model = BiDAFRLGenerator(self.c, self.c_mask, self.ch, self.q, self.q_mask, self.qh, self.a, self.a_mask,
                                         self.ah, self.y1, self.y2, self.word_mat, self.char_mat, self.dropout,
                                         self.N, self.PL, self.QL, self.AL, self.CL, config.hidden, config.char_dim,
                                         config.glove_dim, self.num_words, config.use_pointer, config.attention_tpye,
                                         self.reward, self.sa, config.mixing_ratio, config.pre_step)
                self.loss, self.loss_ml, self.loss_rl = model.build_model(self.global_step)
                self.symbols = model.sample(config.beam_size) if config.baseline_type == "beam" else model.sample_rl()
                self.symbols_rl = model.sample_rl()
                self.lr = tf.cond(self.global_step < config.pre_step, lambda: tf.minimum(config.ml_learning_rate,
                                  0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1)),
                                  lambda: config.rl_learning_rate)
            total_params()

            if config.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.loss += l2_loss

            if config.decay is not None:
                self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.loss = tf.identity(self.loss)

                    self.assign_vars = []
                    for var in tf.global_variables():
                        v = self.var_ema.average(var)
                        if v:
                            self.assign_vars.append(tf.assign(var, v))

            if trainable:
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                        zip(capped_grads, variables), global_step=self.global_step)
