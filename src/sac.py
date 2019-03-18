import numpy as np
import tensorflow as tf
import math

class SAC:

    def __init__(self, config):
        self.epochs = config["epochs"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]

    def input_fn(self, transition_matrices):

        # Fetch current_state, action, reward and next_state matrices.
        _, _, current_states, actions, rewards, next_states = transition_matrices

        current_states = current_states.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # Convert action dtype for indexing.
        actions = actions.astype(np.int32)

        # Split dataset into train and validation set.
        split_percentage = 0.8
        num_samples = len(current_states)
        train_size = int(split_percentage * num_samples)
        valid_size = int((1-split_percentage) * num_samples)
        train_set = (current_states[:train_size], actions[:train_size], rewards[:train_size], next_states[:train_size])
        valid_set = (current_states[-valid_size:], actions[-valid_size:], rewards[-valid_size:], next_states[-valid_size:])

        # Calculate number of train batches.
        self.num_train_batches = int(math.ceil(train_size / float(self.train_batch_size)))
        # Calculate number of valid batches.
        self.num_valid_batches = int(math.ceil(valid_size / float(self.valid_batch_size)))

        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices(train_set).batch(self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_set).batch(self.valid_batch_size)

        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter

    def value_network(self, current_states, variable_scope, trainable):
        """Computes value function at a given state"""
        with tf.variable_scope(variable_scope, trainable=trainable):

            # Value function estimate for the current state.
            v = tf.layers.dense(current_states, 1, activation=tf.nn.relu)
        return v

    def q_network(self, current_states, actions, variable_scope, trainable):
        """Computes the action-value function (Q value) at a given state and
        action"""
        with tf.variable_scope(variable_scope, trainable=trainable):

            # Concatenate current state and action in a vector and pass it to Q
            # network to observe Q value.
            state_action = tf.concat(current_states, actions)
            q = tf.layers.dense(state_action, 1, activation=tf.nn.relu)

            return q

    def policy_network(self, current_states, actions, variable_scope, trainable):
        """Computes probability of taking the action for current_states and given actions"""
        with tf.variable_scope(variable_scope, trainable=trainable):

            # Concatenate current state and action in a vector and pass it to
            # policy network to observe the probability of taking the action in
            # the state.
            state_action = tf.concat(current_states, actions)
            pie = tf.layers.dense(state_action, 1, activation=tf.nn.sigmoid)

            return pie

    def soft_value_function_loss(self, current_states, actions):
        """Computes the loss to update soft value function."""
        with tf.name_scope("value_function_loss"):
            v = self.value_network(current_states,
                                   variable_scope="value_network", trainable=True)
            q = self.q_network(current_states, actions,
                               variable_scope="q_network", trainable=False)
            pie = self.policy_network(current_states, actions,
                                      variable_scope="policy_network", trainable=False)
            soft_v = tf.reduce_sum(q - tf.log(pie))
            v_loss_op = tf.reduce_sum(0.5 * tf.pow((v - soft_v), 2))
            return v_loss_op

    def soft_q_function_loss(self, current_states, actions, rewards, next_states):
        """Computes the loss to update soft Q function."""
        with tf.name_scope("q_function_loss"):
            v_target = self.value_network(next_states,
                                   variable_scope="target_value_network",
                                   trainable=False)
            q = self.q_network(current_states, actions,
                               variable_scope="q_network", trainable=True)
            q_target = rewards + self.gamma * tf.reduce_sum(v_target)
            q_loss_op = tf.reduce_sum(0.5 * tf.pow((q - q_target), 2))
            return q_loss_op

    def train(self, current_states, actions, rewards, next_states):

        # Create loss operation for value function update.
        v_loss_op = self.soft_value_function_loss(current_states, actions)

        # Create loss operation for Q function update.
        q_loss_op = self.soft_q_function_loss(current_states, actions, rewards,
                                             next_states)

        # TODO Create loss operation for policy network update.

        # Combine all the loss operations
        loss = tf.group(v_loss_op, q_loss_op)

        # Create optimization operation.
        optimize_op = self.optimize_fn(loss)

        # Log loss in tensorboard summary.
        mean_loss, mean_loss_update_op = utils.avg_loss(loss)
        tf.summary.scalar('mean_loss', mean_loss)
        tf.summary.scalar('loss', loss)

        # Summaries for all the trainable variables.
        utils.parameter_summaries(tf.trainable_variables())

        # TODO: Add tensorboard model evaluation metrics.

        summary = tf.summary.merge_all()

        return optimize_op, loss, summary


    def copy(self, primary_scope, target_scope):

        with tf.name_scope("copy"):

            primary_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=primary_scope)
            target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)

            primary_variables_sorted = sorted(primary_variables, key=lambda v: v.name)
            target_variables_sorted = sorted(target_variables, key=lambda v: v.name)

            assign_ops = []

            for primary_var, target_var in zip(primary_variables_sorted, target_variables_sorted):
                assign_ops.append(target_var.assign(tf.identity(primary_var)))

            copy_op = tf.group(*assign_ops)

            return copy_op

    def fit(self, transition_matrices, restore=False, global_step=0):

        # Check if the export directory is present,
        # if not present create new directory.
        # if os.path.exists(self.export_dir) and restore is False:
        #     raise ValueError("Export directory already exists. Please specify different export directory.")
        # elif os.path.exists(self.export_dir) and restore:
        #     print ("Restoring model from latest checkpoint.")
        #     pass
        # else:
        #     os.mkdir(self.export_dir)

        # self.builder=tf.saved_model.builder.SavedModelBuilder(self.SERVING_DIR)

        # Save model config
        # params = self.get_params()
        # with open(os.path.join(self.export_dir, 'params.json'), 'wb') as f:
        #     json.dump(params, f)

        print ("transition_matrices: ", transition_matrices.shape)

        # Clear deafult graph stack and reset global graph definition.
        tf.reset_default_graph()

        # Set seed for random.
        tf.set_random_seed(self.seed)

        # Get data iterator ops.
        train_init_op, valid_init_op, data_iter = self.input_fn(transition_matrices)

        # Create iterator.
        current_states, actions, rewards, next_states = data_iter.get_next()

        # Get loss and optimization ops
        optimize_op, loss, summary = self.train(current_states, actions, rewards, next_states)

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            # Initialize variables in graph.
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Restore model checkpoint.
            if restore:
                self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            # Create file writer directory to store summary and events.
            train_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/train', sess.graph)
            valid_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/valid')

            # Create model copy op.
            copy_op = self.copy(primary_scope='primary', target_scope='target')

            # Initialize step count.
            step = global_step
            for epoch in range(self.epochs):

                # Initialize training set iterator.
                sess.run(train_init_op)

                for batch in range(self.num_train_batches):

                    train_loss, train_summary, _ = sess.run([loss, summary, optimize_op])

                    # Log training dataset.
                    train_writer.add_summary(train_summary, step)

                    # Check if step to update Q target.
                    if step % self.update_step == 0:
                        sess.run(copy_op)

                    step +=1

                # Log results every step.
                if epoch % self.log_step == 0:

                    # Get validation set.
                    # Initialize training set iterator.
                    sess.run(valid_init_op)

                    # Get results on validation set.
                    valid_loss, valid_summary = sess.run([loss, summary])

                    # Log validation dataset.
                    valid_writer.add_summary(valid_summary, step)

            # Save model checkpoint.
            self.saver.save(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            return step

    def predict(self, test_X):

        # Clear deafult graph stack and reset global graph definition.
        tf.reset_default_graph()

        # Get data iterator ops.
        # _, _, data_iter = self.input_fn(transition_matrices)

        # Create iterator.
        # current_states, _, _, _ = data_iter.get_next()

        current_states = tf.placeholder(shape=[None, 4], dtype=tf.float32)

        # Get Q value of current state.
        q_primary_logits = self.q_network(current_states, variable_scope="primary", trainable=True)

        # Get index of max q_target_logits.
        action = tf.argmax(q_primary_logits, axis=1)

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restore model checkpoint.
            self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            # Result on test set batch.
            logits_test, action_test = sess.run([q_primary_logits, action], {current_states: test_X.reshape(-1, 4)})

        return logits_test, action_test[0]
