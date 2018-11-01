import tensorflow as tf
import numpy as np

class KABLSTM(object):

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=hidden_units
        n_layers=1
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        print(x)
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        #return outputs[-1]

        # output transformation to the original tensor type
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs
    

    # return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
    def max_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

        # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        output = tf.reshape(output, [-1, width])

        return output

    def avg_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
        
        # do avg-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.avg_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        
        output = tf.reshape(output, [-1, width])
        
        return output

    def kb_module(self, H, ent_emb, ent_W):
        h_h, w = int(H.get_shape()[1]), int(H.get_shape()[2]) #30,64
        print H.get_shape(),ent_emb.get_shape()
        h_e, w_e = int(ent_emb.get_shape()[2]), int(ent_emb.get_shape()[3]) #5

        out1 = tf.reduce_mean(H, axis=1) #(?,64)

        reshape_h1 = tf.expand_dims(out1, 1) #(?,1,64)
        reshape_h1 = tf.expand_dims(reshape_h1, 1) #(?,1,1,64)
        reshape_h1 = tf.tile(reshape_h1, [1, h_h, h_e, 1]) #(?,30,5,64)
        reshape_h1 = tf.reshape(reshape_h1, [-1, w]) #(? * 30 * 5,64)
        reshape_h2 = tf.reshape(ent_emb, [-1, w_e]) #(? * 30 * 5,64)
        print reshape_h1.get_shape(),reshape_h2.get_shape()
        M = tf.tanh(tf.add(tf.matmul(reshape_h1, ent_W['Wqm']), tf.matmul(reshape_h2, ent_W['Wam']))) #(?,att)
        M = tf.matmul(M, ent_W['Wms']) #(?,1)

        S = tf.reshape(M, [-1, h_e]) #(?,5)
        S = tf.nn.softmax(S)

        S_diag = tf.matrix_diag(S) #(?,5,5)
        reshape_ent = tf.reshape(ent_emb, [-1,h_e,w_e]) #(?*30,5,64)
        attention_a = tf.matmul(S_diag, reshape_ent) #(?*30,5,64)
        attention_a = tf.reshape(attention_a, [-1, h_h, h_e, w_e]) #(?,30,5,64)

        out2 = tf.reduce_mean(attention_a, axis=2)

        return tf.concat([H, out2],2),out2
        return H,out2

    def overlap(self, embed1, embed2):
        overlap1 = tf.matmul(embed1,tf.transpose(embed2,[0,2,1]))
        overlap1 = tf.expand_dims(tf.reduce_max(overlap1,axis=2),-1)
        
        overlap2 = tf.matmul(embed2,tf.transpose(embed1,[0,2,1]))
        overlap2 = tf.expand_dims(tf.reduce_max(overlap2,axis=2),-1)
        embed1 = tf.concat([embed1,overlap1],2)
        embed2 = tf.concat([embed2,overlap2],2)
        return embed1,embed2

    def attentive_pooling(self, h1, h2, U):
        dim = int(h1.get_shape()[2])
        transform_left = tf.einsum('ijk,kl->ijl',h1, U)
        att_mat = tf.tanh(tf.matmul(transform_left, tf.transpose(h2,[0,2,1])))
        row_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=1)),-1, name='answer_attention')
        column_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=2)),-1, name='question_attention1')
        out2 = tf.reshape(tf.matmul(tf.transpose(h2,[0,2,1]),row_max),[-1,dim])
        out1 = tf.reshape(tf.matmul(tf.transpose(h1,[0,2,1]),column_max),[-1,dim])
        return out1,out2

    def attentive_combine(self, h1, h2, U1, h3, h4, U2):
        dim1 = int(h1.get_shape()[2])
        dim2 = int(h3.get_shape()[2])
        transform_left = tf.einsum('ijk,kl->ijl',h1, U1)
        att_mat = tf.tanh(tf.matmul(transform_left, tf.transpose(h2,[0,2,1])))
        row_max1 = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=1)),-1, name='answer_attention1')
        column_max1 = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=2)),-1, name='question_attention1')
        transform_left = tf.einsum('ijk,kl->ijl',h3, U2)
        att_mat = tf.tanh(tf.matmul(transform_left, tf.transpose(h4,[0,2,1])))
        row_max2 = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=1)),-1)
        column_max2 = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=2)),-1)
        row_max = tf.tanh(tf.add(row_max1, row_max2, name='answer_attention2'))
        column_max = tf.tanh(tf.add(column_max1, column_max2, name='question_attention2'))
        #row_max = tf.add(row_max1, row_max2, name='answer_attention2')
        #column_max = tf.add(column_max1, column_max2, name='question_attention2')
        out_a1 = tf.reshape(tf.matmul(tf.transpose(h2,[0,2,1]),row_max),[-1,dim1])
        out_q1 = tf.reshape(tf.matmul(tf.transpose(h1,[0,2,1]),column_max),[-1,dim1])
        out_a2 = tf.reshape(tf.matmul(tf.transpose(h4,[0,2,1]),row_max),[-1,dim2])
        out_q2 = tf.reshape(tf.matmul(tf.transpose(h3,[0,2,1]),column_max),[-1,dim2])
        out1 = tf.concat([out_q1,out_q2],1)
        out2 = tf.concat([out_a1,out_a2],1)
        print out1.get_shape(),out2.get_shape()
        return out1,out2


    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, embedding_matrix, entity_embedding_matrix,entity_embedding_dim, entity_vocab_size,n_entity, mode):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.ent_x1 = tf.placeholder(tf.int32, [None, sequence_length, n_entity], name="ent_x1")
        self.ent_x2 = tf.placeholder(tf.int32, [None, sequence_length, n_entity], name="ent_x2")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.add_fea = tf.placeholder(tf.float32, [None, 4], name="add_fea")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0, name="l2_loss")
        print self.ent_x1.get_shape()
        # Embedding layer
        with tf.name_scope("embedding"):
            # char embedding
            if embedding_matrix.all() != None:
                self.W = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
            else:
                self.W = tf.get_variable("emb", [vocab_size, embedding_size])

            # char embedding
            if entity_embedding_matrix.all() == None:
                print 'random embedding'
                self.ent_emb = tf.Variable(tf.random_uniform([entity_vocab_size, entity_embedding_dim], -1.0, 1.0),trainable=False,name="ent_emb")
            else:
                print 'graph embedding'
                self.ent_emb = tf.Variable(entity_embedding_matrix, trainable=False, name="ent_emb", dtype=tf.float32)
            
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_ent1 = tf.nn.embedding_lookup(self.ent_emb, self.ent_x1)
            #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            self.embedded_ent2 = tf.nn.embedding_lookup(self.ent_emb, self.ent_x2)
            #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)\
            self.embedded_chars1,self.embedded_chars2 = self.overlap(self.embedded_chars1,self.embedded_chars2)
            print self.embedded_chars1.get_shape()
            print self.embedded_ent1.get_shape()

        attention_size = 200
        with tf.name_scope("ent_weight"):
            self.ent_W = {
                'Wam' : tf.Variable(tf.random_uniform([entity_embedding_dim,attention_size], -1.0, 1.0), trainable=True, name="ent_Wam"),
                'Wqm' : tf.Variable(tf.random_uniform([2*hidden_units,attention_size], -1.0, 1.0), trainable=True, name="ent_Wqm"),
                'Wms' : tf.Variable(tf.random_uniform([attention_size,1], -1.0, 1.0), trainable=True, name="ent_Wms")
            }

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("RNN"):
            self.h1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size+1, sequence_length, hidden_units)
            self.h2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size+1, sequence_length, hidden_units)
            #self.out1=self.kb_module(self.h1, self.embedded_ent1, self.ent_W)
            #self.out2=self.kb_module(self.h2, self.embedded_ent2, self.ent_W)
            self.out1,self.ent_att1=self.kb_module(self.h1, self.embedded_ent1, self.ent_W)
            self.out2,self.ent_att2=self.kb_module(self.h2, self.embedded_ent2, self.ent_W)



        # Create a convolution + maxpool layer for each filter size
        filter_sizes = [2,3]
        height = int(self.ent_att1.get_shape()[2])
        num_filters = 200
        
        h_left = []
        h_right = []

        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, height, num_filters]
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(
                    self.ent_att1,
                    W,
                    stride=1,
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print h.get_shape()
                h_left.append(h)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(
                    self.ent_att2,
                    W,
                    stride=1,
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_right.append(h)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_q = tf.concat(h_left,2)
        print self.out1.get_shape()
        self.h_a = tf.concat(h_right,2)

        with tf.name_scope("attentive_pooling"):
            U1 = tf.get_variable(
                    "U1",
                    shape=[2*hidden_units, 2*hidden_units],
                    initializer=tf.contrib.layers.xavier_initializer())
            U2 = tf.get_variable(
                    "U2",
                    shape=[num_filters_total, num_filters_total],
                    initializer=tf.contrib.layers.xavier_initializer())

            if mode == 'none':
                self.out1, self.out2 = self.attentive_pooling(self.h1,self.h2,U1)
            else:
                self.out1, self.out2= self.attentive_combine(self.h1,self.h2,U1,self.h_q,self.h_a,U2)
        


        # Compute similarity
        with tf.name_scope("similarity"):
            sim_size = int(self.out1.get_shape()[1])
            W = tf.get_variable(
                "W",
                shape=[sim_size, sim_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.out1, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.out2), 1, keep_dims=True)
            
            #print self.sims

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Make input for classification
        if mode == 'raw':
            self.new_input = tf.concat([self.out1, self.sims, self.out2], 1, name='new_input')
        else:
            self.new_input = tf.concat([self.out1, self.sims, self.out2, self.add_fea], 1, name='new_input')

        num_feature = int(self.new_input.get_shape()[1])
        
        # hidden layer
        hidden_size = 200
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[num_feature, hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            print self.h_drop

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[hidden_size, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.prob = tf.nn.xw_plus_b(self.h_drop, W, b)
            self.soft_prob = tf.nn.softmax(self.prob, name='distance')
            self.predictions = tf.argmax(self.soft_prob, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=tf.one_hot(self.input_y,2))
            self.loss = tf.reduce_mean(losses) 
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())

            self.total_loss = self.loss + self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

