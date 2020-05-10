from sklearn.linear_model import LogisticRegression
from rbm import RBM
from loadMnist import load_data
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from softmax import SoftmaxRegression
from sklearn.decomposition import IncrementalPCA

datasets = load_data(0.2)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]


print('Encoding...')

rbm = RBM(n_visible=28 * 28, n_hidden=100)

rbm.load_parameters()

encoded_train, _ = rbm.sample_hidden_given_visible(train_set_x)
encoded_valid, _ = rbm.sample_hidden_given_visible(valid_set_x)
encoded_test, _ = rbm.sample_hidden_given_visible(test_set_x)
#
# print('Dimensionality reduction...')
# pca_original = IncrementalPCA(n_components = 100)
# n_batches = 100
#
# for x_batch_original in np.array_split(train_set_x, n_batches):
#     pca_original.partial_fit(x_batch_original)
#
# encoded_train = pca_original.transform(train_set_x)
# encoded_valid = pca_original.transform(valid_set_x)
# encoded_test = pca_original.transform(test_set_x)

# encoded_train = train_set_x
# encoded_valid = valid_set_x
# encoded_test = test_set_x

print('Softmax training...')

# softmax_reg = LogisticRegression(multi_class='multinomial', solver='sag', C=10, max_iter= 5000)
# softmax_reg.fit(encoded_train, train_set_y)

softmax_reg = SoftmaxRegression(eta=0.01, epochs=100, minibatches=4800, random_seed=42)
softmax_reg.fit(encoded_train, train_set_y)

# softmax_reg = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm_clf', SVC(kernel='poly', degree=6, coef0=0.2, C=0.5))
# ])
# softmax_reg.fit(encoded_train, train_set_y)

predict_train = softmax_reg.predict(encoded_train)
conf_mx_train_encoded = confusion_matrix(train_set_y, predict_train)
print('Accuracy on train is:', conf_mx_train_encoded.diagonal().sum()/conf_mx_train_encoded.sum())
rows_sum = conf_mx_train_encoded.sum(axis=1, keepdims=True)
conf_mx_train_encoded = conf_mx_train_encoded/rows_sum
df_conf_mx_train_encoded = pd.DataFrame(conf_mx_train_encoded, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_conf_mx_train_encoded, cmap='PRGn', annot=True, annot_kws={"size": 12}) # font size
plt.xlabel('True class')
plt.ylabel('Predicted class')
plt.title('Confusion matrix on train')
plt.ylabel('Predicted class')
plt.title('Confusion matrix on train')
plt.show()
plt.close()
#
predict_test = softmax_reg.predict(encoded_valid)
conf_mx_test_encoded = confusion_matrix(valid_set_y, predict_test)
print('Accuracy on validation is:', conf_mx_test_encoded.diagonal().sum()/conf_mx_test_encoded.sum())
rows_sum = conf_mx_test_encoded.sum(axis=1, keepdims=True)
conf_mx_test_encoded = conf_mx_test_encoded/rows_sum
df_conf_mx_test_encoded = pd.DataFrame(conf_mx_test_encoded, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_conf_mx_test_encoded, cmap='PRGn', annot=True, annot_kws={"size": 12}) # font size
plt.xlabel('True class')
plt.ylabel('Predicted class')
plt.title('Confusion matrix on test')
plt.show()
plt.close()

# pred, true = 9, 7
# errori = train_set_x[(softmax_reg.predict(encoded_train) == pred) & (train_set_y == true)]
# errori_encoded = encoded_train[(softmax_reg.predict(encoded_train) == pred) & (train_set_y == true)]
# probErr, err = rbm.sample_visible_given_hidden(errori_encoded)
# probabilities = softmax_reg.predict_proba(errori_encoded)
# for i in range(5):
#     plt.imshow(errori[i,:].reshape((28, 28)), cmap="gray")
#     plt.show()
#     plt.imshow(probErr[i,:].reshape((28, 28)), cmap="gray")
#     plt.show()
