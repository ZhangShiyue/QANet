import tensorflow as tf
from layers import _linear, total_params, regularizer


class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, trainable=True, graph=None):

        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.N = config.batch_size if trainable else config.test_batch_size
            self.QL = config.ques_limit if trainable else config.test_ques_limit
            self.CL = config.char_limit
            self.NV = len(word_mat)

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.q = tf.placeholder(tf.int32, [self.N, self.QL], "question")
            self.qh = tf.placeholder(tf.int32, [self.N, self.QL, self.CL], "question_char")
            self.q_mask = tf.cast(self.q, tf.bool)
            self.qa_id = tf.placeholder(tf.int32, [self.N], "qa_id")

            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=config.word_trainable)
            self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(config.hidden),
                                                      input_keep_prob=1.0 - self.dropout)
            self.loop_function = self._loop_function
            self.dw = config.glove_dim
            self.d = config.hidden
            self.lr = tf.minimum(config.ml_learning_rate, 0.001 / tf.log(999.) *
                                 tf.log(tf.cast(self.global_step, tf.float32) + 1))

            self.loss = self.build_model(self.global_step)
            self.symbols = self.sample(config.beam_size)

            total_params()

            if config.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.loss += l2_loss

            if trainable:
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                        zip(capped_grads, variables), global_step=self.global_step)

    def build_model(self, global_step):
        outputs, oups = self.decode()
        # compute loss
        batch_loss = self._compute_loss(outputs, oups)
        loss = tf.reduce_mean(batch_loss)
        return loss

    def decode(self, reuse=None):
        with tf.variable_scope("Decoder_Layer", reuse=reuse):
            oups = tf.split(self.q, [1] * self.QL, 1)
            state = self.cell.zero_state(self.N, dtype=tf.float32)
            outputs = []
            for i, inp in enumerate(oups):
                einp = tf.reshape(tf.nn.embedding_lookup(self.word_mat, inp), [self.N, self.dw])
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                h, state = self.cell(einp, state)

                with tf.variable_scope("AttnOutputProjection"):
                    output = _linear([h] + [einp], output_size=self.dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                    outputs.append(output)

            return outputs, oups

    def sample(self, beam_size):
        with tf.variable_scope("Decoder_Layer", reuse=True):
            oups = tf.split(self.q, [1] * self.QL, 1)
            state = self.cell.zero_state(self.N, dtype=tf.float32)
            symbols = []
            prev = None
            prev_probs = tf.zeros((self.N, 1))
            for i, inp in enumerate(oups):
                einp = tf.nn.embedding_lookup(self.word_mat, inp)
                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_probs, index, prev_symbol = self.loop_function(beam_size, prev, prev_probs, i)
                        state = tuple(tf.gather_nd(s, index) for s in state)  # update prev state
                        for j, symbol in enumerate(symbols):
                            symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
                        symbols.append(prev_symbol)

                # update cell state
                h, state = self.cell(tf.reshape(einp, [-1, self.dw]),
                                     tuple(tf.reshape(s, [-1, self.d]) for s in state))
                h = tf.reshape(h, [self.N, -1, self.d])
                state = tuple(tf.reshape(s, [self.N, -1, self.d]) for s in state)

                with tf.variable_scope("AttnOutputProjection"):
                    oinp = tf.reshape(tf.concat([h, einp], -1), [-1, self.d + self.dw])
                    output = _linear([oinp], output_size=self.dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                    output = tf.reshape(output, [self.N, -1, self.dw])

                prev = output

            # process the last symbol
            einp, prev_probs, index, prev_symbol = self.loop_function(beam_size, prev, prev_probs, i)
            for j, symbol in enumerate(symbols):
                symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
            symbols.append(prev_symbol)

            # output the final best result of beam search
            index = tf.stack([tf.range(self.N), tf.zeros(self.N, dtype=tf.int32)], axis=-1)
            for k, symbol in enumerate(symbols):
                symbols[k] = tf.gather_nd(symbol, index)

            return symbols

    def _loop_function(self, beam_size, prev, prev_probs, i):
        dim = 1 if i == 1 else beam_size
        # scatter attention probs
        logit = tf.matmul(tf.reshape(prev, [-1, self.dw]), self.word_mat, transpose_b=True)
        logit = tf.reshape(logit, [self.N, dim, -1])
        dist_g = tf.nn.softmax(logit)
        final_dist = tf.log(dist_g)
        # beam search
        prev_probs = tf.expand_dims(prev_probs, -1)
        prev = final_dist + prev_probs  # batch_size * dim * NVP
        prev = tf.reshape(prev, [self.N, -1])  # batch_size * (dim * NVP)
        probs, prev_symbolb = tf.nn.top_k(prev, beam_size)  # batch_size * beam_size
        index = prev_symbolb // self.NV
        bindex = tf.tile(tf.expand_dims(tf.range(self.N), -1), [1, beam_size])
        index = tf.stack((bindex, index), axis=2)
        prev_symbol = prev_symbolb % self.NV

        # embedding_lookup
        emb_prev = tf.nn.embedding_lookup(self.word_mat, prev_symbol)

        return emb_prev, probs, index, prev_symbol

    def _compute_loss(self, ouputs, oups):
        weights = []
        crossents = []
        for output, oup in zip(ouputs[:-1], oups[1:]):
            logit = tf.matmul(output, self.word_mat, transpose_b=True)
            target = tf.reshape(oup, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            crossents.append(crossent * weight)
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps
