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
iteration = 200
k = np.linspace(1, L_unknown, num=L)
index = np.linspace(1, N, num=N)
Error200_1 = np.zeros(N)
Error200_2 = np.zeros(N)
WeightError200_1 = np.zeros(N)
WeightError200_2 = np.zeros(N)


for itr in range (0, iteration):
    x = np.random.randn(N, 1).flatten()
    w0 = 1 / k * np.exp(-(k - 4) ** 2 / 4)
    d = np.convolve(x, w0, mode="full")[0:200]
    # define adaptation gain factor mu
    Px = sum(x ** 2) / x.shape[0]
    Pnoise = Px/100;
    noise = np.random.normal(0, np.sqrt(Pnoise), 200)
    xnoise = x + noise
    mu = 1 / (3 * L * Px)
    error1, weights1 = lms(xnoise, d, mu, L)
    error2, weights2 = lms(x, d, mu, L)
    WeightError1 = np.zeros(N)
    WeightError2 = np.zeros(N)
    for i in range(L - 1, d.shape[0]):
        WeightError1[i] = compute_error(np.squeeze(np.asarray(weights1[i])), w0)
        WeightError2[i] = compute_error(np.squeeze(np.asarray(weights2[i])), w0)
    Error200_1 += error1
    Error200_2 += error2
    WeightError200_1 +=WeightError1
    WeightError200_2 += WeightError2
    print(itr)

ErrorAve1 = Error200_1/iteration
ErrorAve2 = Error200_2/iteration
WeightErrorAve1 = WeightError200_1/iteration
WeightErrorAve2 = WeightError200_2/iteration

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(index, ErrorAve1**2, label='Input with AWGN', color='purple')
plt.plot(index, ErrorAve2**2, label='Original input', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error')

plt.figure(1)
plt.subplot(2, 2, 3)
plt.semilogy(index, ErrorAve1**2, label='Input with AWGN', color='purple')
plt.semilogy(index, ErrorAve2**2, label='Original input', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error in dB')

plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(index, WeightErrorAve1, label='Input with AWGN', color='purple')
plt.plot(index, WeightErrorAve2, label='Original input', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm')

plt.figure(1)
plt.subplot(2, 2, 4)
plt.semilogy(index, WeightErrorAve1, label='Input with AWGN', color='purple')
plt.semilogy(index, WeightErrorAve2, label='Original input', color='b',ls='--')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm in dB')