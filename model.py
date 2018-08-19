import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention, _linear, multihead_attention


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, rerank=False, graph=None):
        self.config = config
        self.rerank = rerank
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.rerank:
                self.c, self.q, self.a, self.ch, self.qh, self.y1, self.y2, self.qa_id, self.can_id = batch.get_next()
                # self.cv = self.cv + tf.constant([1] * 4 + [0] * (len(word_mat) + config.test_para_limit - 4))
            else:
                self.c, self.q, self.a, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
                # self.cv = self.cv + tf.constant([1] * 4 + [0] * (len(word_mat) + (config.para_limit if trainable else config.test_para_limit) - 4))

            # self.word_unk = tf.get_variable("word_unk", shape=[1, config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
            self.num_voc = len(word_mat)
            # additional_word_mat = tf.tile(tf.nn.embedding_lookup(original_word_mat, [1]),
            #                               [config.para_limit if trainable else config.test_para_limit, 1])
            # self.word_mat = tf.concat([original_word_mat, additional_word_mat], axis=0)

            self.char_mat = tf.get_variable(
                    "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

            # self.num_voc = len(word_mat) + (config.para_limit if trainable else config.test_para_limit)
            self.loop_function = None
            # self.loop_function = None if trainable or self.rerank else \
            #     self._extract_argmax_and_embed(self.word_mat, self.num_voc, config.beam_size, self.c, self.cv)
            # self.pred_loop_function = None if trainable or self.rerank else self._pred_beam_search(config.beam_size)

            self.c_mask = tf.cast(self.c, tf.bool)
            self.q_mask = tf.cast(self.q, tf.bool)
            self.a_mask = tf.cast(self.a, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            self.a_len = tf.reduce_sum(tf.cast(self.a_mask, tf.int32), axis=1)

            if trainable:
                self.c_maxlen, self.q_maxlen, self.a_maxlen = config.para_limit, config.ques_limit, config.ans_limit
            else:
                self.c_maxlen, self.q_maxlen, self.a_maxlen = config.test_para_limit, config.test_ques_limit, config.test_ans_limit

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
                capped_grads, _ = tf.clip_by_global_norm(
                        gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                        zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, AL, CL, d, dc, nh, dw = config.test_batch_size if self.loop_function or self.rerank else config.batch_size, \
                                       self.c_maxlen, self.q_maxlen, self.a_maxlen, \
                                       config.char_limit, config.hidden, config.char_dim, config.num_heads, config.glove_dim

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
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Encoder"):
            c = residual_block(c_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.c_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.c_len,
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
                               seq_len=self.q_len,
                               scope="Input_Encoder_Block",
                               reuse=True,  # Share the weights between passage and question
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)

            with tf.variable_scope("BiDAF"):
                # BiDAF
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

            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]
            # for i in range(3):
            #     if i % 2 == 0:  # dropout every 2 blocks
            #         self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
            self.enc.append(residual_block(self.enc[0],
                           num_blocks=7,
                           num_conv_layers=2,
                           kernel_size=5,
                           mask=self.c_mask,
                           num_filters=d,
                           num_heads=nh,
                           seq_len=self.c_len,
                           scope="Model_Encoder",
                           bias=False,
                           # reuse=True if i > 0 else None,
                           dropout=self.dropout))

        with tf.variable_scope("Decoder"):
            # self-attention
            self.dec = residual_block(tf.nn.embedding_lookup(self.word_mat, self.a),
                               num_blocks=7,
                               num_conv_layers=0,
                               kernel_size=0,
                               mask=self.a_mask,
                               causality=True,
                               num_filters=d,
                               num_heads=nh,
                               scope="Output_Encoder_Block",
                               bias=False,
                               dropout=self.dropout,
                               input_projection=True)

            # attend to memory
            self.dec = residual_block(self.dec,
                                    memory=self.enc[1],
                                    num_blocks=7,
                                    num_conv_layers=0,
                                    kernel_size=0,
                                    mask=self.c_mask,
                                    num_filters=d,
                                    num_heads=nh,
                                    scope="Memory_Attention_Block",
                                    bias=False,
                                    dropout=self.dropout)

            # out projection
            self.dec = conv(self.dec, dw, name="out_proj")

            # output
            logits = tf.matmul(tf.reshape(self.dec, [-1, dw]), self.word_mat, transpose_b=True)

            # loss
            weight = tf.reshape(tf.cast(self.a_mask, tf.float32) * tf.constant([0.] + [1.] * (AL - 1)), [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.a, [-1])) * weight
            self.loss = tf.reduce_sum(loss) / (tf.reduce_sum(weight) + 1e-12)

        # with tf.variable_scope("Output_Layer"):
        #     start_logits = tf.squeeze(
        #         conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"), -1)
        #     end_logits = tf.squeeze(
        #         conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"), -1)
        #     self.logits = [mask_logits(start_logits, mask=self.c_mask),
        #                    mask_logits(end_logits, mask=self.c_mask)]
        #     logits1, logits2 = [l for l in self.logits]
        #     outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
        #                       tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        #     outer = tf.matrix_band_part(outer, 0, config.ans_limit)
        #     self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        #     self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        #     if self.pred_loop_function:
        #         self.byp1, self.byp2 = self.pred_loop_function(outer)
        #     losses = tf.nn.softmax_cross_entropy_with_logits(
        #             logits=logits1, labels=self.y1)
        #     losses2 = tf.nn.softmax_cross_entropy_with_logits(
        #             logits=logits2, labels=self.y2)
        #     self.loss = tf.reduce_mean(losses + losses2)


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

    def _pred_beam_search(self, beam_size):
        def loop_function(outer):
            PL = outer.get_shape()[1].value
            bprobs1, byp1 = tf.nn.top_k(tf.log(tf.reduce_max(outer, axis=2)), k=beam_size)
            bprobs2 = tf.tile(tf.log(tf.reduce_max(outer, axis=1)), [beam_size, 1]) + tf.transpose(bprobs1)
            bprobs2 = tf.reshape(bprobs2, [1, PL * beam_size])
            bprobs2, bindex = tf.nn.top_k(bprobs2, k=beam_size)
            bindex = tf.squeeze(bindex, 0)
            index = bindex // PL
            byp2 = bindex % PL
            byp1 = tf.gather(tf.squeeze(byp1, 0), index)
            return byp1, byp2
        return loop_function

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
            logit = tf.matmul(output, self.word_mat, transpose_b=True) * tf.to_float(self.cv)
            dist_g = tf.nn.softmax(logit)
            final_dist = p_gen * dist_g + (1 - p_gen) * dist_c
            # get loss
            indices = tf.concat((batch_nums, oup), axis=1)
            gold_probs = tf.gather_nd(final_dist, indices)
            crossent = -tf.log(tf.clip_by_value(gold_probs,1e-10,1.0))
            target = tf.reshape(oup, [-1])
            weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            crossents.append(crossent * weight)
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps, tf.reduce_sum(log_perps) / tf.cast(batch_size, tf.float32)

    def _extract_argmax_and_embed(self, embedding, num_symbols, beam_size, c, cv, update_embedding=True):
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
            # prev = tf.matmul(prev, embedding, transpose_b=True)
            # prev = prev * tf.to_float(cv)
            # prev = tf.log(tf.nn.softmax(prev))
            batch_size = prev.get_shape()[0].value
            PL = c.get_shape()[1].value
            bc = tf.tile(c, [batch_size, 1])
            batch_nums_c = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, PL])
            indices_c = tf.stack((batch_nums_c, bc), axis=2)
            dist_c = tf.scatter_nd(indices_c, attn_w, [batch_size, num_symbols])
            logit = tf.matmul(prev, embedding, transpose_b=True) * tf.to_float(cv)
            dist_g = tf.nn.softmax(logit)
            final_dist = p_gen * dist_g + (1 - p_gen) * dist_c

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