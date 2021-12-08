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
L_set = [5, 9, 11]
N = 200
k = np.linspace(1, L_unknown, num=L_unknown)
index = np.linspace(1, N, num=N)

# define signal and desired signal
x = np.random.randn(N, 1).flatten()
w0 = 1/k*np.exp(-(k-4)**2/4)
d = np.convolve(x, w0, mode="full")[0:200]
# define adaptation gain factor mu
Px = sum(x**2)/x.shape[0]
error = np.zeros([3,N])
WeightError = np.zeros([3,N])
mu = np.zeros(3)

for j in range (0,3):
    mu[j] = 1 / (3 * L_set[j] * Px)
    erroritr, weightsitr = lms(x, d, mu[j], L_set[j])
    error[j] = erroritr

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(index, error[0]**2, label='L = 5', color='purple')
plt.plot(index, error[1]**2, label='L = 9', color='blue')
plt.plot(index, error[2]**2, label='L = 11', color='yellow')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error')

plt.figure(1)
plt.subplot(1, 2, 2)
plt.semilogy(index, error[0]**2, label='L = 5', color='purple')
plt.semilogy(index, error[1]**2, label='L = 9', color='blue')
plt.semilogy(index, error[2]**2, label='L = 11', color='yellow')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error in dB')

print('The adaptation gain is',"%.3f"%mu[0], "%.3f"%mu[1],"%.3f"%mu[2],'for length = 5, 9 and 11, respectively')

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