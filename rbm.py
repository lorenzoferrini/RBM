import numpy
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvas

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

class RBM(object):
    def __init__(self, n_visible=28*28, n_hidden=100,
                 W=None, biasV=None, biasH=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.randSeed = numpy.random.RandomState(42)

        if W is None:
            W = numpy.asarray(self.randSeed.normal(0,0.01, size = (n_visible, n_hidden)))

        if biasV is None:
            biasV = numpy.asarray(self.randSeed.uniform(0, 1, size = (n_hidden))) #TODO set biasV to log[pi/(1−pi)] where pi is the proportionof training vectors in which unit i is on

        if biasH is None:
            biasH = numpy.zeros(n_visible)

        self.W = W
        self.biasV = biasV
        self.biasH = biasH

    def _forwardProp(self, visible):
        arg = visible.dot(self.W) + self.biasV
        return sigmoid(arg)

    def sample_hidden_given_visible(self, visible):
        pH_V = self._forwardProp(visible)
        hidden = numpy.greater(pH_V, self.randSeed.uniform(0,1)).astype(numpy.int)
        return pH_V, hidden

    def _backwardProp(self, hidden):
        arg =  hidden.dot(self.W.T) + self.biasH
        return sigmoid(arg)

    def sample_visible_given_hidden(self, hidden):
        pV_H = self._backwardProp(hidden)
        visible = numpy.greater(pV_H, self.randSeed.uniform(0,1)).astype(numpy.int)
        return pV_H, visible

    def weight_update(self, input, lr=0.1, k=1, useProb=1):

        batch_size = input.shape[0]
        lr = lr/batch_size
        pH_V_0, hidden_0 = self.sample_hidden_given_visible(input)
        visible, hidden, pH_V = input, hidden_0, pH_V_0

        for i in range(k):
            pV_H, visible = self.sample_visible_given_hidden(hidden)
            pH_V, hidden = self.sample_hidden_given_visible(pV_H)
        if useProb:
            self.W += lr * (numpy.dot(input.T, pH_V_0) - numpy.dot(visible.T, pH_V))
            self.biasH += lr * numpy.mean(input - visible, axis=0)
            self.biasV += lr * numpy.mean(pH_V_0 - pH_V, axis=0)
        else:
            self.W += lr * (numpy.dot(input.T, hidden_0) - numpy.dot(visible.T, hidden))
            self.biasH += lr * numpy.mean(input - visible, axis=0)
            self.biasV += lr * numpy.mean(pH_V_0 - pH_V, axis=0)

    def free_energy(self, v):
        ''' Compute the free energy for a visible state'''
        vbias_term = numpy.sum(numpy.dot(v, self.biasH.T))
        x_b = numpy.dot(v, self.W) + self.biasV
        hidden_term = numpy.sum(numpy.log(1 + numpy.exp(x_b)))
        return numpy.mean(- hidden_term - vbias_term)

    def reconstruction_error(self, v):
        _, hidden = self.sample_hidden_given_visible(v)
        visible, _ = self.sample_visible_given_hidden(hidden)
        reconErr = numpy.mean(numpy.square(v - visible))
        return reconErr


    def plot_weights(self):
        fig = plt.figure(figsize= (14, 14), dpi=100)
        for i in range(self.n_hidden):
            lin = numpy.sqrt(self.n_hidden)
            fig.add_subplot(numpy.floor(lin), numpy.ceil(lin), i + 1)
            plt.axis('off')
            plt.imshow(self.W[:,i].reshape((28, 28)), cmap="gray")
        plt.savefig('weights.png')
        plt.show()

    def load_parameters(self):
        print('Loading parameters...')
        W = numpy.genfromtxt('W.csv', delimiter=',')
        # W = W.reshape((self.n_visible, self.n_hidden))
        biasH = numpy.genfromtxt('biasH.csv', delimiter=',')
        # biasH = biasH.reshape((self.n_visible,1))
        biasV = numpy.genfromtxt('biasV.csv', delimiter=',')
        # biasV = biasV.reshape((self.n_hidden,1))
        self.W = W
        self.biasH = biasH
        self.biasV = biasV

    def save_params(self, prefix=''):
        print('Saving parameters...')
        numpy.savetxt(prefix+'W.csv', self.W, delimiter=',')
        numpy.savetxt(prefix+'biasV.csv', self.biasV, delimiter=',')
        numpy.savetxt(prefix+'biasH.csv', self.biasH, delimiter=',')


    def activation_clip(self, input, filename='outpy.avi'):
        print('Saving clip...')
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 2, (1920,1080))
        for i in range(input.shape[0]):
            pH_V, hidden = self.sample_hidden_given_visible(input[i,:])
            pV_H, visible = self. sample_visible_given_hidden(hidden)
            frame = numpy.ones((1080,1920,3), dtype='float32')
            fig = plt.figure(figsize=(14, 14), dpi=100)
            canvas = FigureCanvas(fig)
            for j in range(hidden.size):
                lin = numpy.sqrt(hidden.size)
                fig.add_subplot(numpy.floor(lin), numpy.ceil(lin), j + 1)
                plt.axis('off')
                if hidden[j]:
                    plt.imshow(self.W[:, j].reshape((28, 28)), cmap="hot")
                else:
                    plt.imshow(self.W[:, j].reshape((28, 28)), cmap="gray")
            canvas.draw()
            w_image = numpy.array(canvas.renderer.buffer_rgba())
            w_image = cv2.cvtColor(w_image, cv2.COLOR_RGBA2BGR)
            w_image = cv2.resize(w_image, (1080,1080), interpolation=cv2.INTER_NEAREST)
            source = cv2.resize(numpy.ceil(input[i,:].reshape((28,28))*255), (420,420), interpolation=cv2.INTER_NEAREST)
            visible = visible.astype('float32')
            dest = cv2.resize(numpy.ceil(pV_H.reshape((28,28))*255), (420,420), interpolation=cv2.INTER_NEAREST)
            frame[:,420:1500,:] = w_image[:,:,:3].astype('float32')
            plt.close('all')
            for i in range(3):
                frame[330:330+420,0:420,i] = source
                frame[330:330+420, 1500:1920, i] = dest

            out.write(frame.astype('uint8'))
        out.release()
