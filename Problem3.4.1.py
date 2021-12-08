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
iteration = 200
k = np.linspace(1, L_unknown, num=L)
index = np.linspace(1, N, num=N)
forgetting_set = [0.99,0.95,0.8]
ErrorAve = np.zeros([3,N])
WeightErrorAve = np.zeros([3,N])


for muitr in range (0,3):
    Error200 = np.zeros(N)
    WeightError200 = np.zeros(N)
    for itr in range(0, iteration):
        x = np.random.randn(N, 1).flatten()
        w0 = 1 / k * np.exp(-(k - 4) ** 2 / 4)
        d = np.convolve(x, w0, mode="full")[0:200]
        weights, error = rls(x, d, L, forgetting_set[muitr])
        WeightError = np.zeros(N)
        for i in range(L - 1, d.shape[0]):
            WeightError[i] = compute_norm_error(np.squeeze(np.asarray(weights[i])), w0)
        Error200 += error
        WeightError200 += WeightError
        print(itr)
    ErrorAve[muitr] = Error200 / iteration
    WeightErrorAve[muitr] = WeightError200 / iteration

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(index, ErrorAve[0]**2, label='forget_factor = 0.99', color='purple')
plt.plot(index, ErrorAve[1]**2, label='forget_factor = 0.95', color='blue')
plt.plot(index, ErrorAve[2]**2, label='forget_factor = 0.8', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error')

plt.figure(1)
plt.subplot(2, 2, 3)
plt.semilogy(index, ErrorAve[0]**2, label='forget_factor = 0.99', color='purple')
plt.semilogy(index, ErrorAve[1]**2, label='forget_factor = 0.95', color='blue')
plt.semilogy(index, ErrorAve[2]**2, label='forget_factor = 0.8', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error in dB')

plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(index, WeightErrorAve[0], label='forget_factor = 0.99', color='purple')
plt.plot(index, WeightErrorAve[1], label='forget_factor = 0.95', color='b')
plt.plot(index, WeightErrorAve[2], label='forget_factor = 0.8', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm')

plt.figure(1)
plt.subplot(2, 2, 4)
plt.semilogy(index, WeightErrorAve[0], label='forget_factor = 0.99', color='purple')
plt.semilogy(index, WeightErrorAve[1], label='forget_factor = 0.95', color='b')
plt.semilogy(index, WeightErrorAve[2], label='forget_factor = 0.8', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm in dB')