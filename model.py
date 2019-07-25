from mynn.layers.dense import dense


class Model:
    def __init__(self):
        self.dense1 = dense(512, 50)

    def __call__(self, x):
        ''' Forward data through the network.

        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D)
            The data to forward through the network.

        Returns
        -------
        mygrad.Tensor, shape=(N, 1)
            The model outputs.
        '''
        return self.dense1(x)

    @property
    def parameters(self):
        ''' A convenience function for getting all the parameters of our model. '''
        return self.dense1.parameters


from mynn.optimizers.sgd import SGD
model=Model()
optim = SGD(model.parameters, learning_rate=0.1)
for epoch_cnt in range(100):
    good_image = np.
    text =
    bad_image =
    sim_to_good = sim(se_text(text), model(good_image))
    sim_to_bad = sim(se_text(text), model(bad_image))

    # compute the loss associated with our predictions(use softmax_cross_entropy)
    loss = margin_ranking_loss(sim_good, sim_bad, 1, 0.1)

    # back-propagate through your computational graph through your loss
    loss.backward()

    # compute the accuracy between the prediction and the truth
    acc = accuracy(prediction, truth)

    # execute gradient descent by calling step() of optim
    optim.step()

    # null your gradients
    loss.null_gradients()