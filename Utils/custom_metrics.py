import numpy as np

import matplotlib.pyplot as plt

def mean_relative_l2_error(true_data, pred_data):
    # Returns relative l2-norm error as defined in paper https://arxiv.org/pdf/2203.00360.pdf,
    # Non-linear manifold ROM with Convolutional Autoencoders and Reduced Over-Collocation method

    N=true_data.shape[0]
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    err_denom=np.linalg.norm(true_data, ord=2, axis=1)
    
    return np.exp(np.sum(np.log(err_numer/err_denom))/N)
    # return np.sum(err_numer/err_denom)/N

""" def mean_relative_l2_error(true_data, pred_data):
    # Returns relative l2-norm error as defined in paper https://arxiv.org/pdf/2203.00360.pdf,
    # Non-linear manifold ROM with Convolutional Autoencoders and Reduced Over-Collocation method
    print('MEAN REL L2 ERROR')
    print('Size of matrices: ', true_data.shape, pred_data.shape)
    N=true_data.shape[0]
    print('N: ', N)
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)

    # print('NUMERATOR ERROR L2', np.sum(err_numer)/N)
    err_denom=np.linalg.norm(true_data, ord=2, axis=1)
    # print('DENOM ERROR L2', np.sum(err_denom)/N)
    print('List of rel errors: ', err_numer/err_denom)
    print('Sum: ', np.sum(err_numer/err_denom))

    plt.plot(err_numer)
    # plt.semilogy()
    # plt.show()
    plt.plot(err_denom)
    print(true_data[0])
    print(np.sqrt(np.sum(np.power(true_data[0], 2))))
    # plt.plot(np.max(np.abs(true_data-pred_data), axis=1))
    # plt.plot(np.max(np.abs(true_data), axis=1))
    # plt.plot(np.min(np.abs(true_data-pred_data), axis=1))
    # plt.plot(np.min(np.abs(true_data), axis=1))
    # plt.plot(err_numer/err_denom)
    # plt.plot(np.ones((3000))*np.sum(err_numer/err_denom)/N)
    # plt.plot(np.ones((3000))*np.linalg.norm(true_data-pred_data))
    # plt.plot(np.ones((3000))*np.linalg.norm(pred_data))
    plt.semilogy()
    plt.show()
    
    return np.sum(err_numer/err_denom)/N """

def relative_forbenius_error(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data)
    # print('NUMERATOR ERROR FROB', err_numer)
    err_denom=np.linalg.norm(true_data)
    # print('DENOM ERROR FROB', err_denom)
    # return np.log(err_numer/err_denom)
    return err_numer/err_denom

def relative_l2_error_list(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    err_denom=np.linalg.norm(true_data, ord=2, axis=1)
    # plt.plot(err_numer/err_denom)
    # plt.plot(np.ones((300))*np.exp(np.sum(np.log(err_numer/err_denom))/true_data.shape[0]))
    # print(np.exp(np.sum(np.log(err_numer/err_denom))/true_data.shape[0]))
    # plt.semilogy()
    # plt.show()
    return err_numer/err_denom

def l2_error_list(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    # plt.plot(err_numer)
    # plt.plot(np.ones((300))*np.exp(np.sum(np.log(err_numer))/true_data.shape[0]))
    # plt.semilogy()
    # plt.show()
    return err_numer

def mean_l2_error(true_data, pred_data):
    # Returns relative l2-norm error as defined in paper https://arxiv.org/pdf/2203.00360.pdf,
    # Non-linear manifold ROM with Convolutional Autoencoders and Reduced Over-Collocation method
    N=true_data.shape[0]
    err_numer=np.linalg.norm(true_data-pred_data, ord=2, axis=1)
    return np.exp(np.sum(np.log(err_numer))/N)
    # return np.sum(err_numer)/N

def forbenius_error(true_data, pred_data):
    err_numer=np.linalg.norm(true_data-pred_data)
    # return np.log(err_numer)
    return err_numer