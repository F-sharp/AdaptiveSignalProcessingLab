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
L_unknown = 9  # length of unknown system IR
L = 9
N = 400
iteration = 200
k = np.linspace(1, L_unknown, num=L)
index = np.linspace(1, N, num=N)
forget_factor = 0.8

Error200lms =np.zeros(N)
Error200nlms =np.zeros(N)
Error200rls =np.zeros(N)
WeightError200lms =np.zeros(N)
WeightError200nlms =np.zeros(N)
WeightError200rls =np.zeros(N)

for itr in range (0, iteration):
    x = np.random.randn(N, 1).flatten()
    w1 = 1 / k * np.exp(-(k - 4) ** 2 / 4)
    w2 = 1 / k * np.exp(-(k - 2) **3 / 3)
    d1 = np.convolve(x[0:200], w1, mode="full")[0:200]
    d2 = np.convolve(x[200:400], w2, mode="full")[0:200]
    d = np.concatenate((d1, d2))
    # define adaptation gain factor mu
    Px = sum(x ** 2) / x.shape[0]
    mu = 1 / (3 * L * Px)
    errorlms, weightslms = lms(x, d, mu, L)
    errornlms, weightsnlms = nlms(x, d, mu, L)
    weightsrls, errorrls = rls(x, d, L, forget_factor)

    WeightErrorlms = np.zeros(N)
    WeightErrornlms = np.zeros(N)
    WeightErrorrls = np.zeros(N)
    for i in range(L - 1, 200):
        WeightErrorlms[i] = compute_norm_error(np.squeeze(np.asarray(weightslms[i])), w1)
        WeightErrornlms[i] = compute_norm_error(np.squeeze(np.asarray(weightsnlms[i])), w1)
        WeightErrorrls[i] = compute_norm_error(np.squeeze(np.asarray(weightsrls[i])), w1)
    for i in range(200, 400):
        WeightErrorlms[i] = compute_norm_error(np.squeeze(np.asarray(weightslms[i])), w2)
        WeightErrornlms[i] = compute_norm_error(np.squeeze(np.asarray(weightsnlms[i])), w2)
        WeightErrorrls[i] = compute_norm_error(np.squeeze(np.asarray(weightsrls[i])), w2)

    Error200lms += errorlms
    Error200nlms += errornlms
    Error200rls += errorrls
    WeightError200lms +=WeightErrorlms
    WeightError200nlms += WeightErrornlms
    WeightError200rls += WeightErrorrls
    print(itr)

ErrorAve = np.zeros([3,N])
WeightErrorAve = np.zeros([3,N])

ErrorAve[0] = Error200lms/iteration
ErrorAve[1] = Error200nlms/iteration
ErrorAve[2] = Error200rls/iteration
WeightErrorAve[0] = WeightError200lms/iteration
WeightErrorAve[1] = WeightError200nlms/iteration
WeightErrorAve[2] = WeightError200rls/iteration

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(index, ErrorAve[0]**2, label='LMS', color='purple')
plt.plot(index, ErrorAve[1]**2, label='NLMS', color='blue')
plt.plot(index, ErrorAve[2]**2, label='RLS', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error')

plt.figure(1)
plt.subplot(2, 2, 3)
plt.semilogy(index, ErrorAve[0]**2, label='LMS', color='purple')
plt.semilogy(index, ErrorAve[1]**2, label='NLMS', color='blue')
plt.semilogy(index, ErrorAve[2]**2, label='RLS', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('error in dB')

plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(index, WeightErrorAve[0], label='LMS', color='purple')
plt.plot(index, WeightErrorAve[1], label='NLMS', color='b')
plt.plot(index, WeightErrorAve[2], label='RLS', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm')

plt.figure(1)
plt.subplot(2, 2, 4)
plt.semilogy(index, WeightErrorAve[0], label='LMS', color='purple')
plt.semilogy(index, WeightErrorAve[1], label='NLMS', color='b')
plt.semilogy(index, WeightErrorAve[2], label='RLS', color='r')
plt.legend()
plt.xlabel('sample number k')
plt.ylabel('weight error vector norm in dB')