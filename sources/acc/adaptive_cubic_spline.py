'''
Locally Adjusted Cubic-Spline Capping for MODIS data
Reference: Chen, Jing M., Feng Deng, and Mingzhen Chen. "Locally adjusted cubic-spline capping for reconstructing seasonal trajectories of a satellite-derived surface parameter." IEEE Transactions on Geoscience and Remote Sensing 44.8 (2006): 2230.
Author: Yunan Luo
Date: Oct 28, 2016
License: BSD 3

Input: Time series of one pixel
Output: Daily values of interpolated data
'''
import numpy as np

class Spline(object):
	def __init__(self, num_iter = 5):
		self.a = None
		self.b = None
		self.c = None
		self.d = None
		self.x = None
		self.num_iter = num_iter 
		# number of iterations, 3 is suggested in the reference, 5 is used here

	def GUCC(self, x, y, Gamma, lamb = 0.5):
		h = np.diff(x)
		p = np.array([2*(h[i] + h[i+1]) for i in range(h.shape[0] - 1)])
		r = 3.0 / h
		f = np.array([-(r[i] + r[i+1]) for i in range(r.shape[0] - 1)])
		N = x.shape[0]

		M = np.diag(p[0:N-2])
		rng = np.arange(N-3)
		M[rng, rng + 1] = h[1:N-2]
		M[rng + 1, rng] = h[1:N-2]

		Q_trans = np.zeros((N-2, N))
		for i in range(Q_trans.shape[0]):
			Q_trans[i, i]   = r[i]
			Q_trans[i, i+1] = f[i]
			Q_trans[i, i+2] = r[i+1]
		Q = Q_trans.T

		mu = 2. * (1 - lamb) / (3 * lamb)

		b_tmp = np.linalg.solve(M + mu * np.dot(np.dot(Q.T, Gamma), Q), np.dot(Q.T, y))
		self.b = np.zeros(N)
		self.b[1:N-1] = b_tmp

		self.d = y - mu * np.dot(np.dot(Gamma, Q), b_tmp)	

		self.a = np.zeros(N)
		self.c = np.zeros(N)
		for i in range(N-1):
			self.a[i] = (self.b[i+1] - self.b[i]) / (3. * h[i])
			self.c[i] = (self.d[i+1] - self.d[i]) / (1. * h[i]) - h[i] * (self.b[i+1] + 2*self.b[i]) / 3.

	def LACC(self, x, y):
		N = x.shape[0]
		h = np.diff(x)
		y_1st = np.zeros(N)		# first derivative
		for i in range(1, N - 1):
			y_1st[i] = (y[i+1] - y[i-1]) / (h[i] + h[i-1])
		y_2nd = np.zeros(N)		# second derivative
		for i in range(1, N - 1):
			y_2nd[i] = (y_1st[i+1] - y_1st[i-1]) / (h[i] + h[i-1])
		y_2nd_max = np.max(y_2nd)

		gamma = 1 - (np.minimum(abs(y_2nd), y_2nd_max) * 1.0 / y_2nd_max) ** (1 / 2.5)
		Gamma = np.diag(gamma)
		self.GUCC(x, y, Gamma = Gamma, lamb = 0.5)	


	def fit(self, x, y):
		self.x = np.array(x)
		for _ in range(self.num_iter):
			self.GUCC(x, y, Gamma = np.identity(len(x)), lamb = 0.5)
			y_hat = self.predict(x)
			y = np.maximum(y, y_hat)
			self.LACC(x, y)

	def splines_func(self, x):
		i = -1	#not assigned
		for j in range(self.x.shape[0] - 1):
			if self.x[j] <= x and self.x[j+1] > x:
				i = j
				break
		if i == -1:
			i = self.x.shape[0] - 1
		
		y_hat = self.a[i] * (x - self.x[i])**3 \
				+ self.b[i] * (x - self.x[i])**2 \
				+ self.c[i] * (x - self.x[i]) \
				+ self.d[i]
		return y_hat
		
	def predict(self, x_list):
		y_list = []
		for x in x_list:
			y_list.append(self.splines_func(x))
		y_list = np.array(y_list)
		return y_list
		
if __name__ == "__main__":
        import matplotlib
        # Force matplotlib to not use any Xwindows backend.
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
	# x: day of year
	# y: value	
	x = np.load('2012_nir_x.npy')
	y = np.load('2012_nir_y.npy')

	spline = Spline()
	spline.fit(x, y)
	x_pred = range(x.min(), x.max() + 1) # interpolate at daily level
	y_pred = spline.predict(x_pred)

	fig = plt.figure(figsize=(5,5))	
	ax = fig.add_subplot(1,1,1)
	ax.plot(x, y, 'r.', markersize=5, label=u'Observations')	
	ax.plot(x_pred ,y_pred, color='b', label='spline')	
	ax.set_xlabel('Day of Year')
	ax.set_ylabel('NIR')
	ax.set_title('Year 2012')
	ax.legend(loc='upper right', fontsize=10)
	plt.tight_layout()
	plt.savefig('output.pdf')
