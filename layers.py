'''
In the Siamese network, we defined some custom layers to 
calculate the L1 distance between the two embeddings.

So when we call this model, we need to import these custom layers
'''


# Custom L1 distance layer


# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
         super(L1Dist, self).__init__(**kwargs)
    
    def call(self,input_embedding, validation_embedding):
        
        # Convert inputs to tensors otherwise will meet error: unsupported operand type(s) for -: 'List' and 'List'
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        input_embedding = tf.squeeze(input_embedding, axis=0)  # Remove potential first dimension
        validation_embedding = tf.squeeze(validation_embedding, axis=0)

        # Calculate and return the L1 distance
        return tf.math.abs(input_embedding - validation_embedding)