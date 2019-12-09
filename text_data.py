import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

# Random matrix R generating function
# k is the target dimension, d is the original dimension
def generate_R(k, d):
	# Initialize the matrix
	R = np.zeros(k*d)
	R = R.reshape((k, d))
	for i in range(1, k):
		for j in range(1, d):
			rdnm_number = random.random()
			if rdnm_number < 1/6:
				R[i][j] = float(1)
			elif rdnm_number > 5/6:
				R[i][j] = float(-1)
	R = R*np.sqrt(3)
	return R

# Compute the inner product
def get_inner_product(X):
	[row, col] = X.shape
	prod = np.zeros(col * (col - 1) // 2)
	index = 0
	for i in range(col):
		for j in range(col):
			if j < i:
				prod[index] = np.inner(X[:,i], X[:,j])
				index += 1
	return prod

# Define the original dimension
n = int()
m = int(5000)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# categories = ['sci.med']

textdata = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
word_counts = count_vect.fit_transform(textdata.data)

tf_transformer = TfidfTransformer(use_idf=False).fit(word_counts)
# the results is a scipy.sparse.csr.csr_matrix
X = tf_transformer.transform(word_counts)
# Clip the dataset
X = X[:,:m]
# normalize and transpose the dataset
X = normalize(X, 'l2')
X = X.asfptype()
X = X.transpose()
print("The text data fequency matrix size is:", X.shape)
prod_init = get_inner_product(scipy.eye(m)*X)
[dummy, n] = X.shape

test_dimensions = np.linspace(30, 690, 22).astype(int)
prods = np.zeros(n * (n - 1) // 2 * len(test_dimensions)).reshape((len(test_dimensions), n * (n - 1) // 2))
for i, k in enumerate(test_dimensions): #Test with final dimensions 0 to 600
	# Define the random matrix R
	R = generate_R(k, m)
	print("The random matrix size is:", R.shape)
	# Multipy the random matrix with A
	X_hat = R*X*np.sqrt(float(m)/float(k))
	X_hat = normalize(X_hat.transpose(), 'l2').transpose()
	print("The projected matrix size is:", X_hat.shape)
	# Compute inner product
	prods[i] = get_inner_product(X_hat)

avg_err = np.zeros(len(test_dimensions))
for i in range(len(test_dimensions)):
	err = np.absolute(prods[i] - prod_init)
	avg_err[i] = np.average(err)

# Expected eps for each dimension
eps = np.sqrt(np.divide(np.log10(n), test_dimensions.astype(float) / 8))
int_eps = np.sqrt(np.divide(np.log10(n), test_dimensions.astype(float)))

upper_bound = np.zeros(len(test_dimensions))
interpolated = np.zeros(len(test_dimensions))
for i, e in enumerate(eps):
	tmp_upper = prod_init * e
	tmp_interp = prod_init * int_eps[i]
	upper_bound[i] = np.average(tmp_upper)
	interpolated[i] = np.average(tmp_interp)

# Plot the results
plt.rcParams.update({'font.size':36})
plt.figure(figsize=(18,12))
plt.plot(test_dimensions, avg_err, label="Practical error", marker="*", markersize=20)
# Compute the confidence interval
plt.plot(test_dimensions, upper_bound, linewidth=2, label="Upper bound $\sqrt{8(\log n)/k}$\n confidence 93.0%")
# plt.plot(test_dimensions, interpolated, label="$\sqrt{4(\log n)/k}$")
plt.plot(test_dimensions, interpolated, linewidth=2, label="$\sqrt{(\log n)/k}$")
plt.xlabel("Target dimension")
plt.ylabel("Average Error")
plt.legend(frameon=False)
plt.show()
