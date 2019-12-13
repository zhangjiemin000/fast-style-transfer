# Get the gradients of the generated image wrt the loss
import numpy as np
import keras.backend as K




class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None
        self.combination_image = None
        self.loss_value = None
        self.img_width = None
        self.img_height=None
        self.fetch_loss_and_grads = None

    def setup(self,combination_image,loss,img_width,img_height):
        self.combination_image = combination_image
        self.loss_value = loss
        self.loss_lose = None
        self.img_width = img_width
        self.img_height = img_height
        grads = K.gradients(self.loss_value, self.combination_image)[0]
        # Function to fetch the values of the current loss and the current gradients
        self.fetch_loss_and_grads = K.function([self.combination_image], [self.loss_value, grads])

    def loss(self, x):
        assert self.loss_lose is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values