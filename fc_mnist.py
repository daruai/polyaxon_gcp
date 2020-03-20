import argparse
import tensorflow as tf
from polyaxon_client.tracking import Experiment

experiment = Experiment()

flatten_list = lambda l: [item for sublist in l for item in sublist]



# Defining the model class

class fc_model:
    '''
    Creates a fully connected model
    '''
    
    def __init__(self, num_neurons, dropout, lr, optimizer):
        tf.random.set_seed(0)
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.lr = lr
        self.optimizer = optimizer
        
        self.model = self.build_model()
        
    
    def get_optimizer(self):
        '''
        Returns the optimizer with the learning rate defined in the init
        '''
        if self.optimizer == 'Adam':
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        elif self.optimizer == 'Adadelta':
            return tf.keras.optimizers.Adadelta(learning_rate=self.lr)
        
        elif self.optimizer == 'Adagrad':
            return tf.keras.optimizers.Adagrad(learning_rate=self.lr)
        
        elif self.optimizer == 'RMSprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        
        elif self.optimizer == 'SGD':
            return tf.keras.optimizers.SGD(learning_rate=self.lr)
        
    def get_callbacks(self):
        '''
        Returns the training callbacks
        '''
        return[tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)]
        
        
    def build_model(self):
        '''
        Builds the model
        '''
        
        print("Building the model...")
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        
        for num in self.num_neurons:
            model.add(tf.keras.layers.Dense(num, activation='relu'))
            if self.dropout:
                model.add(tf.keras.layers.Dropout(self.dropout))
                
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(optimizer=self.get_optimizer(),
                     loss = 'sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.summary()
        
        return model
    
    def train(self, x, y, epochs=1000):
        print("Training...")
        self.model.fit(x, y, epochs=epochs, callbacks=self.get_callbacks())
        
    def evaluate(self, x, y):
        metrics = self.model.evaluate(x, y)
        print("Evaluation results: \n\tloss: {}\n\taccuracy: {}".format(metrics[0], metrics[1]))

def main():
    parser = argparse.ArgumentParser(description="Training a fully connected neural network for mnist")

    parser.add_argument('--num_neurons', action='store',
                    type=int, nargs='*',
                    help="Examples: --num_neurons 20 10")
    parser.add_argument("--dropout", type=float, help="Examples: --dropout 0.2")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer",type=str)
    args = parser.parse_args()
    
    print(args)
    
    # Loading data
    print("Loading data...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = fc_model(num_neurons=args.num_neurons, dropout=args.dropout, lr = args.lr, optimizer = args.optimizer)

    model.train(x_train, y_train)

    model.evaluate(x_test, y_test)
    
    
if __name__ == "__main__":
    main()