import numpy as np
import math

# checked
def data_generation(N, w, std = 0.1):               # Generate a dataset (X,Y) where Y = Xw + e and e is Gaussian with standard deviation = std
    X = np.random.rand(N,2)*2-1                     # X 是Nx2的矩阵，实际数据里是100x2
    err_set = np.random.randn(N,1)*std              # w 是2x1，所以Y是100x1， err_set也是100x1
    Y = np.dot(X, w) + err_set
    Y = np.clip(Y, -1., 1.)                         # 小于-1的全部换成-1，大于1的全部换成1
    return (X, Y)

# checked
def L2_loss(X, Y, w):
    tmp = np.dot(X, w) - Y                          # 100 x 2 x 2 x 1 - 100 x 1 tmp = 100 x 1 
    return np.dot(tmp.T, tmp)                       # 1 x 100 x 100 x 1 = 1 

# checked
def sigmasq_func(eps, delta, sens = 1.):            # Compute the variance for a (eps, delta)-DP Gaussian mechanism with sensitivity = sens
    return 2.*np.log(1.25/delta)*sens**2/(eps**2)

#checked
def comp_reverse(eps, delta, T):                    # Given the privacy parameter of the composed mechanism, compute the privacy parameter of each sub-mechanism (by either composition or advanced composition)
    return eps/T, delta/T

def LR_GD(X, Y, eps, delta, T, C = 1., eta = 0.1):  # Solve the Linear regression with (eps, delta)-differentially private SGD
    N, d = X.shape                                  # Get the dimension of X, here N = 100, d = 2, X:100x2, Y:100x1
    w = np.zeros((d,1))                             # w 是2x1的00矩阵
    # 计算参数
    eps_u, delta_u = comp_reverse(eps, delta, T)    # Compute the privacy parameter of each update, (eps_u, delta_u), given (eps, delta, T)
    sigmasq = sigmasq_func(eps, delta)              # Compute the variance when sensitivity = 1
    L = 0.01 * N
    print("σ square:  " + str(sigmasq))

    for i in range(T):
        tmp = np.dot(X,w)-Y                         # tmp = Xw - Y           
        gradient = 2*np.dot(X.T, tmp)               # Compute the gradient g = 2 * X^T * (Xw - Y), g now 2 x 1
        
        # to do: Clip gradient
        for grad_item in gradient:
            grad_item[0] = grad_item[0] / (max (1, math.sqrt((grad_item[0]) ** 2) / C))
        # print(gradient)
        # to do: Add noise
        sum = 0
        index = 0
        for grad_item in gradient:
            sum += gradient[index][0]
            index += 1
        gradient_temp = []
        sigma_sq_dis = sigmasq * (C ** 2)
        norm_dis = np.random.normal(0.0, sigma_sq_dis, d)
        # print(norm_dis)
        for i in range(d):
            gradient_temp.append((1 / L) * (sum + norm_dis[i]))
        gradient_temp_toarr = [[item] for item in gradient_temp]
        gradient_temp_arr = np.array(gradient_temp_toarr)
        # to do: Gradient decent
        w = w - eta * gradient_temp_arr
        # print(w)
    return w

def LR_FM(X, Y, eps, delta):
    N, d = X.shape                                  # Get the dimension of X, here d = 2
    sens = 2.*(1+d)**2
    sigmasq = sigmasq_func(eps, delta, sens)        # Variance in Functional Mechanism
    noise_1 = np.random.randn(d,d)
    noise_1 *= np.sqrt(sigmasq)/2.
    noise_1 = np.triu(noise_1)
    noise_1 = noise_1 + noise_1.T                   # Compute the noise matrix for X^T*X
    noise_2 = np.random.randn(d,1)
    noise_2 *= np.sqrt(sigmasq)                     # Compute the noise matrix for X^T*Y
    Phi = np.dot(X.T, X)                            # Phi = X^T * X (Phi hat 2 x 2)
    Phi_hat = Phi + noise_1
    # print(type(Phi_hat))
    # print(Phi_hat)
    Identity_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    Phi_hat += Identity_matrix
    XY = np.dot(X.T, Y)
    XY_hat = XY + noise_2
    tmp = np.linalg.inv(Phi_hat)
    w = np.dot(tmp, XY_hat)                         # w = (X^T*X)^(-1) * X^TY
    return w

if (__name__=='__main__'):
    eps, delta = 8, 10**(-7)                        # parameters for DP
    print("ε: " + str(eps))
    print("δ: " + str(delta))
    N, T = 100, 10000
    print(" T: " + str(T))
    w = np.array([[0.5],[0.5]])                     # parameters for LR w ==> 2 x 1矩阵
    X, Y = data_generation(N, w)                    # generate the training data

    # w = LR_GD(X, Y, eps, delta, T)
    # print(L2_loss(X, Y, w))
    w = LR_FM(X, Y, eps, delta)
    print(L2_loss(X, Y, w))