import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention, _linear, multihead_attention


class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, trainable=True, rerank=False, graph=None):
        self.config = config
        self.rerank = rerank
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.c = tf.placeholder(tf.int32,
                                    [None, config.para_limit if trainable and (not rerank) else config.test_para_limit],
                                    "context")
            self.q = tf.placeholder(tf.int32,
                                    [None, config.ques_limit if trainable and (not rerank) else config.test_ques_limit],
                                    "question")
            self.a = tf.placeholder(tf.int32,
                                    [None, config.ans_limit if trainable and (not rerank) else config.test_ans_limit],
                                    "answer")
            self.ch = tf.placeholder(tf.int32,
                                     [None, config.para_limit if trainable and (not rerank) else config.test_para_limit,
                                      config.char_limit], "context_char")
            self.qh = tf.placeholder(tf.int32,
                                     [None, config.ques_limit if trainable and (not rerank) else config.test_ques_limit,
                                      config.char_limit], "question_char")
            self.batch_size = config.batch_size if trainable and (not rerank) else config.test_batch_size
            self.qa_id = tf.placeholder(tf.int32, [self.batch_size], "qa_id")

            if rerank:
                self.can_id = tf.placeholder(tf.int32, [self.batch_size], "can_id")

            # self.word_unk = tf.get_variable("word_unk", shape=[1, config.glove_dim], initializer=initializer())
            original_word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                                trainable=False)
            additional_word_mat = tf.tile(tf.nn.embedding_lookup(original_word_mat, [1]),
                                          [config.para_limit if trainable and (not rerank) else config.test_para_limit,
                                           1])
            self.word_mat = tf.concat([original_word_mat, additional_word_mat], axis=0)
            self.num_voc = len(word_mat) + (config.para_limit if trainable and (not rerank) else config.test_para_limit)

            self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            self.cell = tf.nn.rnn_cell.LSTMCell(config.hidden)
            self.loop_function = None if trainable or rerank \
                else self._extract_argmax_and_embed(self.word_mat, self.num_voc, config.beam_size, self.c)

            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.a_mask = tf.cast(self.a, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

            if trainable and (not rerank):
                self.c_maxlen, self.q_maxlen, self.a_maxlen = config.para_limit, config.ques_limit, config.ans_limit
            else:
                self.c_maxlen, self.q_maxlen, self.a_maxlen = config.test_para_limit, config.test_ques_limit, config.ans_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                    tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                    tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate,
                                     0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                        zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, CL, d, dc, nh, dw, NV = self.batch_size, self.c_maxlen, self.q_maxlen, \
                                           config.char_limit, config.hidden, config.char_dim, config.num_heads, \
                                           config.glove_dim, self.num_voc

        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, dc,
                          bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, dc,
                          bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, qh_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Input_Encoder_Layer"):
            c = residual_block(c_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.c_mask,
                               num_filters=d,
                               num_heads=nh,
                               scope="Input_Encoder_Block",
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)
            q = residual_block(q_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.q_mask,
                               num_filters=d,
                               num_heads=nh,
                               scope="Input_Encoder_Block",
                               reuse=True,  # Share the weights with passage and answer
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)

        with tf.variable_scope("BiDAF"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(residual_block(self.enc[i],
                                               num_blocks=7,
                                               num_conv_layers=2,
                                               kernel_size=5,
                                               mask=self.c_mask,
                                               num_filters=d,
                                               num_heads=nh,
                                               seq_len=self.c_len,
                                               scope="Model_Encoder",
                                               bias=False,
                                               reuse=True if i > 0 else False,
                                               dropout=self.dropout))

        with tf.variable_scope("Decoder_Layer"):
            memory = tf.concat([self.enc[1], self.enc[2], self.enc[3]], axis=-1)
            oups = tf.split(self.a, [1] * self.a_maxlen, 1)
            h = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=d, bias=False, scope="h_initial"))
            c = tf.tanh(_linear(tf.reduce_mean(memory, axis=1), output_size=d, bias=False, scope="c_initial"))
            state = (c, h)
            prev, attn_w, p_gen = None, None, None
            prev_probs = [0.0]
            symbols = []
            attn_ws = []
            p_gens = []
            outputs = []
            for i, inp in enumerate(oups):
                einp = tf.reshape(tf.nn.embedding_lookup(self.word_mat, inp), [N, dw])
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                if self.loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_probs, index, prev_symbol = self.loop_function(prev, attn_w, p_gen, prev_probs, i)
                        h = tf.gather(h, index)  # update prev state
                        state = tuple(tf.gather(s, index) for s in state)  # update prev state
                        for j, symbol in enumerate(symbols):
                            symbols[j] = tf.gather(symbol, index)  # update prev symbols
                        for j, output in enumerate(outputs):
                            outputs[j] = tf.gather(output, index)  # update prev outputs
                        for j, attn_w in enumerate(attn_ws):
                            attn_ws[j] = tf.gather(attn_w, index)  # update prev attn_ws
                        for j, p_gen in enumerate(p_gens):
                            p_gens[j] = tf.gather(p_gen, index)  # update prev p_gens
                        symbols.append(prev_symbol)

                attn, attn_w = multihead_attention(tf.expand_dims(h, 1), units=d, num_heads=1, memory=memory,
                                                   mask=self.c_mask, bias=False,
                                                   is_training=False if self.loop_function is not None else True,
                                                   return_weights=True)

                attn_w = tf.reshape(attn_w, [-1, PL])
                attn_ws.append(attn_w)
                # update cell state
                attn = tf.reshape(attn, [-1, d])
                cinp = tf.concat([einp, attn], 1)
                h, state = self.cell(cinp, state)

                with tf.variable_scope("AttnOutputProjection"):
                    # generation prob
                    p_gen = tf.sigmoid(_linear([h] + [cinp], output_size=1, bias=True, scope="gen_prob"))
                    p_gens.append(p_gen)
                    # generation
                    output = _linear([h] + [cinp], output_size=dw * 2, bias=False, scope="output")
                    output = tf.reshape(output, [-1, dw, 2])
                    output = tf.reduce_max(output, 2)  # maxout
                    outputs.append(output)

                if self.loop_function is not None:
                    prev = output

            if self.loop_function is not None:
                # process the last symbol
                einp, prev_probs, index, prev_symbol = self.loop_function(prev, attn_w, p_gen, prev_probs, i + 1)
                for j, symbol in enumerate(symbols):
                    symbols[j] = tf.gather(symbol, index)  # update prev symbols
                for j, output in enumerate(outputs):
                    outputs[j] = tf.gather(output, index)  # update prev outputs
                for j, attn_w in enumerate(attn_ws):
                    attn_ws[j] = tf.gather(attn_w, index)  # update prev attn_ws
                for j, p_gen in enumerate(p_gens):
                    p_gens[j] = tf.gather(p_gen, index)  # update prev p_gens
                symbols.append(prev_symbol)

                # output the final best result of beam search
                for k, symbol in enumerate(symbols):
                    symbols[k] = tf.gather(symbol, 0)
                for k, output in enumerate(outputs):
                    outputs[k] = tf.expand_dims(tf.gather(output, 0), 0)
                for k, attn_w in enumerate(attn_ws):
                    attn_ws[k] = tf.expand_dims(tf.gather(attn_w, 0), 0)
                for k, p_gen in enumerate(p_gens):
                    p_gens[k] = tf.expand_dims(tf.gather(p_gen, 0), 0)

            self.prev_probs = prev_probs
            self.batch_loss, self.gen_loss = self._compute_loss(outputs, oups, attn_ws, p_gens)
            self.symbols = symbols

        self.loss = self.gen_loss

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

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def _compute_loss(self, ouputs, oups, attn_ws, p_gens):
        batch_size = ouputs[0].get_shape()[0].value
        PL = self.c.get_shape()[1].value
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, PL])
        indices_c = tf.stack((batch_nums_c, self.c), axis=2)
        batch_nums = tf.expand_dims(tf.range(batch_size), 1)
        weights = []
        crossents = []
        for output, oup, attn_w, p_gen in zip(ouputs[:-1], oups[1:], attn_ws[:-1], p_gens[:-1]):
            # combine copy and generation probs
            dist_c = tf.scatter_nd(indices_c, attn_w, [batch_size, self.num_voc])
            logit = tf.matmul(output, self.word_mat, transpose_b=True)
            dist_g = tf.nn.softmax(logit)
            final_dist = p_gen * dist_g + (1 - p_gen) * dist_c
            # get loss
            indices = tf.concat((batch_nums, oup), axis=1)
            gold_probs = tf.gather_nd(final_dist, indices)
            # crossent1 = -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0))
            target = tf.reshape(oup, [-1])
            # crossent0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            crossent = tf.cond(self.global_step < 10000,
                               lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target),
                               lambda: -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0)))
            weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            crossents.append(crossent * weight)
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps, tf.reduce_sum(log_perps) / tf.cast(batch_size, tf.float32)

    def _extract_argmax_and_embed(self, embedding, num_symbols, beam_size, c, update_embedding=True):
        """Get a loop_function that extracts the previous symbol and embeds it.

        Args:
          embedding: embedding tensor for symbols.
          num_symbols: the size of target vocabulary
          update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

        Returns:
          A loop function.
        """

        def loop_function(prev, attn_w, p_gen, prev_probs, _):
            batch_size = prev.get_shape()[0].value
            PL = c.get_shape()[1].value
            bc = tf.tile(c, [batch_size, 1])
            batch_nums_c = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, PL])
            indices_c = tf.stack((batch_nums_c, bc), axis=2)
            dist_c = tf.scatter_nd(indices_c, attn_w, [batch_size, num_symbols])
            logit = tf.matmul(prev, embedding, transpose_b=True)
            dist_g = tf.nn.softmax(logit)
            final_dist = tf.log(p_gen * dist_g + (1 - p_gen) * dist_c)

            # beam search
            prev = tf.nn.bias_add(tf.transpose(final_dist), prev_probs)  # num_symbols*BEAM_SIZE
            prev = tf.transpose(prev)
            prev = tf.expand_dims(tf.reshape(prev, [-1]), 0)  # 1*(BEAM_SIZE*num_symbols)
            probs, prev_symbolb = tf.nn.top_k(prev, beam_size)
            probs = tf.squeeze(probs, [0])  # BEAM_SIZE,
            prev_symbolb = tf.squeeze(prev_symbolb, [0])  # BEAM_SIZE,
            index = prev_symbolb // num_symbols
            prev_symbol = prev_symbolb % num_symbols

            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = tf.stop_gradient(emb_prev)
            return emb_prev, probs, index, prev_symbol

        return loop_function
