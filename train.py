import time
import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
from loadMnist import load_data

# def train(learning_rate=0.1, k=1, training_epochs=15):
plt.close('all')
learning_rate_0 = 0.1
decay = 0.5
k=2
training_epochs=50001
batch_size = 32
print('... loading data')

datasets = load_data()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]


print('... modeling')

rbm = RBM(n_visible=28 * 28, n_hidden=100)

indeces_train = rbm.randSeed.randint(train_set_x.shape[0], size=200)
indeces_valid = rbm.randSeed.randint(valid_set_x.shape[0], size=200)

print('START TRAINING:')

start_time = time.time()
cost_v = np.zeros((int(training_epochs/20+1),1), np.float)
energy_v = np.zeros((int(training_epochs/20+1),1), np.float)
diff_v = np.zeros((int(training_epochs/20+1),1), np.float)

for epoch in range(training_epochs):
    start = (epoch*batch_size) % train_set_x.shape[0]
    end = ((epoch+1)*batch_size) % train_set_x.shape[0]
    if start > end:
        end = train_set_x.shape[0] -1
    batch = train_set_x[start:end, :]
    learning_rate = learning_rate_0 * np.exp(-decay*epoch/training_epochs)
    rbm.weight_update(lr=learning_rate, k=k, input=batch)
    if epoch % 20 == 0:
        indeces_train = rbm.randSeed.randint(train_set_x.shape[0], size=200)
        indeces_valid = rbm.randSeed.randint(valid_set_x.shape[0], size=200)
        cost = rbm.reconstruction_error(valid_set_x[indeces_valid, :])
        energy_valid = rbm.free_energy(valid_set_x[indeces_valid,:])
        energy_train = rbm.free_energy(train_set_x[indeces_train,:])
        idx = np.floor_divide(epoch,20)
        cost_v[idx] = cost
        energy_v[idx] = energy_valid
        diff_v[idx] = energy_train-energy_valid
    if epoch % 500 == 0:
      print('Epoch  %d      Cost:%4.4f  Free Energy:%4.4f   Differece in FE:%4.4f' % (epoch, cost, energy_valid, energy_train-energy_valid))

end_time = time.time()
pretraining_time = (end_time - start_time)

print ('Training took %f ' % int(np.floor_divide(pretraining_time, 60.)),':', int(pretraining_time % 60))

index = 0
probabH, prova_hidden= rbm.sample_hidden_given_visible(train_set_x[index,:])
probabV, prova= rbm.sample_visible_given_hidden(prova_hidden)
plt.imshow(train_set_x[index, :].reshape((28, 28)), cmap="gray")
plt.show()
plt.imshow(probabV.reshape((28, 28)), cmap="gray")
plt.show()

plt.plot(np.linspace(0, energy_v.size, energy_v.size), energy_v)
plt.show()

plt.plot(np.linspace(0, diff_v.size, diff_v.size), diff_v)
plt.show()

print('If you want to save parameters use rbm.save_params()')
# index = rbm.randSeed.randint(valid_set_x.shape[0], size=50)
# rbm.activation_clip(valid_set_x[index,:])
# rbm.activation_clip(valid_set_x[:50,:])
# if __name__ == '__main__':
#   train()