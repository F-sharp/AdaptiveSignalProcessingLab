import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def compute_error(w_k, w_o):
    numerator = np.sum((w_k - w_o) ** 2)
    denominator = np.sum(w_o ** 2)
    return numerator / denominator

def lms(x, d, mu, L):
    N = x.shape[0]
    weights = np.mat(np.zeros([N,L]))
    SignalIn = np.zeros(L)
    error = np.zeros(N)

    for i in range(L - 1, d.shape[0]):
        if i == L - 1:
            w = np.zeros(L)
        DesiredSignal = d[i]
        for j in range(0, L):
            SignalIn[j] = x[i - j]
        y = sum(w * SignalIn)
        error[i] = DesiredSignal - y
        weights[i] = w
        w = w + 2 * mu * error[i] * SignalIn
    return error, weights


# define basic parameters
L_unknown = 9  # length of unknown system IR
L = 9
N = 200
k = np.linspace(1, L_unknown, num=L_unknown)
index = np.linspace(1, N, num=N)
index1 = np.linspace(1,L, num=L)

# define signal and desired signal
x = np.random.randn(N, 1).flatten()
w0 = 1/k*np.exp(-(k-4)**2/4)
d = np.convolve(x, w0, mode="full")[0:200]
# define adaptation gain factor mu
Px = sum(x**2)/x.shape[0]
mu = 1/(3*L*Px)

error, weights = lms(x, d, mu, L)

WeightError = np.zeros(N)

for i in range(L - 1, d.shape[0]):
    WeightError[i] = compute_error(np.squeeze(np.asarray(weights[i])),w0)


plt.figure(1)
plt.plot(index1, np.squeeze(np.asarray(weights[199])),label = 'Final adaptive filter weight')
plt.plot(index1, w0,ls='--',label='Original weight w0')
plt.title('Plot of the final weights of adaptive filter')
plt.legend()
# plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.plot(index, error**2, label='Squared error', color='purple')
# plt.legend()
# plt.xlabel('sample number k')
# plt.ylabel('error')
#
# plt.figure(1)
# plt.subplot(2, 2, 3)
# plt.semilogy(index, error**2, label='Squared error', color='purple')
# plt.legend()
# plt.xlabel('sample number k')
# plt.ylabel('error in dB')
#
# plt.figure(1)
# plt.subplot(2, 2, 2)
# plt.plot(index, WeightError, label='weight error vector norm', color='purple')
# plt.legend()
# plt.xlabel('sample number k')
# plt.ylabel('weight error vector norm')
#
# plt.figure(1)
# plt.subplot(2, 2, 4)
# plt.semilogy(index, WeightError, label='weight error vector norm', color='purple')
# plt.legend()
# plt.xlabel('sample number k')
# plt.ylabel('weight error vector norm in dB')