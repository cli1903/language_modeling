import tensorflow as tf
import numpy as np
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence,
        Feel free to initialize any variables that you find necessary in the constructor.

        vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size

        self.vocab_size = vocab_size
        self.embedding_size = 100
        self.batch_size = 128

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal(shape = [self.vocab_size, self.embedding_size], stddev = 0.1))
        self.b = tf.Variable(tf.random.truncated_normal(shape = [self.vocab_size], stddev = 0.1))
        self.W = tf.Variable(tf.random.truncated_normal(shape = [self.embedding_size * 2, self.vocab_size], stddev = 0.1))
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)


    def call(self,inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)

        :return: prbs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        embedding = tf.reshape(tf.nn.embedding_lookup(self.E, inputs), shape = [-1, self.embedding_size * 2])
        logits = tf.matmul(embedding, self.W) + self.b
        return tf.nn.softmax(logits)

    def loss_function(self,logits,labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param prbs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        print(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits)))
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    
    num_ex = len(train_labels)
    rand_ind = tf.random.shuffle(range(num_ex))
    rand_inp = tf.gather(train_input, rand_ind)
    rand_lab = tf.gather(train_labels, rand_ind)
    for i in range(0, num_ex, model.batch_size):
        #print("batch " + str(i))
        with tf.GradientTape() as tape:
            probs = model.call(rand_inp[:model.batch_size])
            loss = model.loss_function(probs, rand_lab[:model.batch_size])
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            rand_inp = rand_inp[model.batch_size:]
            rand_lab = rand_lab[model.batch_size:]
    
    
    
    

def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)

    :returns: the perplexity of the test set
    """
    prob = model.call(test_input)
    loss = model.loss_function(prob, test_labels)
    return np.exp(loss)

def generate_sentence(word1, word2, length, vocab,model):
    """
    Given initial 2 words, print out predicted sentence of target length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    output_string = np.zeros((1,length), dtype=np.int)
    output_string[:,:2] = vocab[word1], vocab[word2]

    for end in range(2,length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:,start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]
    
    print(" ".join(text))

def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    train_inp, test_inp, word_dict = get_data('data/train.txt', 'data/test.txt')

    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = []
    num_words = len(train_inp)
    for i, word in enumerate(train_inp):
        if i + 1 < num_words - 1:
            train_inputs.append([word, train_inp[i + 1]])
    train_labels = train_inp[2:]
    
    test_inputs = []
    num_words = len(test_inp)
    for i, word in enumerate(test_inp):
        if i + 1 < num_words - 1:
            test_inputs.append([word, test_inp[i + 1]])
    test_labels = test_inp[2:]
    

    # TODO: initialize model and tensorflow variables
    vocab_size = len(word_dict)
    model = Model(vocab_size)

    # TODO: Set-up the training step
    print("train")
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    print("test")
    perp = test(model, test_inputs, test_labels)

    # Print out perplexity 
    print(perp)
    
    generate_sentence('this', 'is', 20, word_dict, model)
    
if __name__ == '__main__':
    main()
