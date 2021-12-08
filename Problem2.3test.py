import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io


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


def nlms(x, d, mu, L):
    StableConstant = 1e-7
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
        w = w + 2 * mu/(StableConstant+np.sum(SignalIn ** 2)) * error[i] * SignalIn
    return error, weights


# define basic parameters
s1 = scipy.io.loadmat('s1.mat')
L_unknown = 9  # length of unknown system IR
L = 9
N = 200
x = np.random.randn(N, 1).flatten()

k = np.linspace(1, L_unknown, num=L)
index = np.linspace(1, N, num=N)

# define desired signal
w0 = 1/k*np.exp(-(k-4)**2/4)
d = np.convolve(x, w0, mode="full")[0:x.shape[0]]
# define adaptation gain factor mu
Px = sum(x**2)/x.shape[0]
muNLMS = 1/(3*L*Px)
muLMS = 1/(10*L*Px)


error1, weights1 = nlms(x, d, muNLMS, L)
error2, weights2 = lms(x, d, muLMS, L)

WeightError1 = np.zeros(N)
WeightError2 = np.zeros(N)

for i in range(L - 1, d.shape[0]):
    WeightError1[i] = compute_error(np.squeeze(np.asarray(weights1[i])),w0)
    WeightError2[i] = compute_error(np.squeeze(np.asarray(weights2[i])), w0)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(index, error1 ** 2, label='NLMS', color='purple')
plt.plot(index, error2 ** 2, label='LMS', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error')

plt.figure(1)
plt.subplot(2, 2, 3)
plt.semilogy(index, error1 ** 2, label='NLMS', color='purple')
plt.semilogy(index, error2 ** 2, label='LMS', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error in dB')
plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(index, WeightError1, label='NLMS', color='purple')
plt.plot(index, WeightError2, label='LMS', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm')

plt.figure(1)
plt.subplot(2, 2, 4)
plt.semilogy(index, WeightError1, label='NLMS', color='purple')
plt.semilogy(index, WeightError2, label='LMS', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm in dB')