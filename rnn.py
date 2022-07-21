import tensorflow as tf
import numpy as np
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 300
        self.batch_size = 150
        self.rnn_size = 300

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        #self.E = tf.Variable(tf.random.truncated_normal(shape = [self.vocab_size, self.embedding_size], stddev = 0.1))
        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = tf.keras.layers.LSTM(self.rnn_size, return_sequences = True, return_state = True)
        #self.dense1 = tf.keras.layers.Dense(20, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)


    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
       
        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        #print(type(inputs))
        embedding = self.E(inputs)
        #embedding = tf.nn.embedding_lookup(self.E, inputs)
        if initial_state:
            rnn, state1, state2 = self.rnn(embedding, initial_state = initial_state)
        else:
            rnn, state1, state2 = self.rnn(embedding)
        #d1 = self.dense1(rnn)
        d2 = self.dense2(rnn)
        return d2, (state1, state2)



    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        print(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits)))
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    num_ex = len(train_labels)
    num_words = model.batch_size * model.window_size
    for i in range(0, num_ex, num_words):
        if i + num_words > num_ex:
            break
        else:
            inputs = tf.reshape(train_inputs[i: i+num_words], shape = [model.batch_size, model.window_size])
            labels = tf.reshape(train_labels[i: i+num_words], shape = [model.batch_size, model.window_size])
            
            with tf.GradientTape() as tape:
                probs, state = model.call(inputs, None)
                loss = model.loss(probs, labels)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples
    
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    
    num_ex = len(test_labels)
    num_words = model.batch_size * model.window_size
    tot_loss = 0
    num_batches = 0.0
    
    for i in range(0, num_ex, num_words):
        if i + num_words > num_ex:
            break
        else:
            inputs = tf.reshape(test_inputs[i: i+num_words], shape = [model.batch_size, model.window_size])
            labels = tf.reshape(test_labels[i: i+num_words], shape = [model.batch_size, model.window_size])
            prob, state = model.call(inputs, None)
            tot_loss += model.loss(prob, labels)
            num_batches += 1
            
    return np.exp(tot_loss/num_batches)


def generate_sentence(word1, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?
    
    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        print(type(next_input))
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.
    
    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs, test_inputs, word_dict = get_data('data/train.txt', 'data/test.txt')
    
    train_labels = train_inputs[1:]
    train_inputs = train_inputs[:-1]
    
    
    test_labels = test_inputs[1:]
    test_inputs = test_inputs[:-1]
    

    # TODO: initialize model and tensorflow variables
    vocab_size = len(word_dict)
    model = Model(vocab_size)

    # TODO: Set-up the training step
    print("train")
    train(model, train_inputs, train_labels)
    
    

    # TODO: Set up the testing steps
    print("test")
    perp = test(model, test_inputs, test_labels)
    
    #generate_sentence("this", 2, word_dict, model)

    # Print out perplexity 
    print(perp)

    
if __name__ == '__main__':
    main()
