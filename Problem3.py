import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compute_norm_error(w_k, w_o):
    numerator = np.sum((w_k - w_o) ** 2)
    denominator = np.sum(w_o ** 2)
    return numerator / denominator

def compute_error(x, d, w):
    return (d - np.dot(w, x))


def compute_kalman_gain(x, last_rinv, forget_factor):
    x = x[:, np.newaxis]
    a = np.matmul(last_rinv, x)
    return a / (forget_factor + np.matmul(x.T, a))


def compute_rinv(x, last_rinv, kalman_gain, forget_factor):
    x = x[:, np.newaxis]
    a = np.matmul(x.T, last_rinv)
    b = np.matmul(kalman_gain, a)
    return (1 / forget_factor) * (last_rinv - b)

def rls(x, d, L, forget_factor):
    N = x.shape[0]
    weights = np.mat(np.zeros([N,L]))
    SignalIn = np.zeros(L)
    error = np.zeros(N)
    rinv = np.eye(L)
    for i in range(L - 1, d.shape[0]):
        if i == L - 1:
            w = np.zeros(L)
        DesiredSignal = d[i]
        for j in range(0, L):
            SignalIn[j] = x[i - j]
        error[i] = compute_error(SignalIn, DesiredSignal, w)
        kalman_gain = compute_kalman_gain(SignalIn, rinv, forget_factor)
        weights[i] = w
        w = w + kalman_gain.flatten() * error[i]
        rinv = compute_rinv(SignalIn, rinv, kalman_gain, forget_factor)

    return np.array(weights), np.array(error)



# define basic parameters
L_unknown = 9  # length of unknown system IR
L = 9
N = 200
k = np.linspace(1, L_unknown, num=L)
index = np.linspace(1, N, num=N)
index1 = np.linspace(1, L, num=L)
forget_factor = 1

# define signal and desired signal
x = np.random.randn(N, 1).flatten()
w0 = 1/k*np.exp(-(k-4)**2/4)
d = np.convolve(x, w0, mode="full")[0:200]

weights, errors = rls(x, d, L, forget_factor)

WeightError = np.zeros(N)

for i in range(L - 1, d.shape[0]):
    WeightError[i] = compute_norm_error(weights[i-L+1],w0)

plt.figure(1)
plt.plot(index1, np.squeeze(np.asarray(weights[199])),label = 'Final adaptive filter weight from RLS')
plt.plot(index1, w0,ls='--',label='Original weight w0')
plt.title('Plot of the final RLS weights of adaptive filter')
plt.legend()

# plt.figure(1)
# plt.subplot(2, 2, 1)
# plt.plot(index, errors**2, label='Squared error', color='purple')
# plt.legend()
# plt.xlabel('sample number k')
# plt.ylabel('error')
#
# plt.figure(1)
# plt.subplot(2, 2, 3)
# plt.semilogy(index, errors**2, label='Squared error', color='purple')
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