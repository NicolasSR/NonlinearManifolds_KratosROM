import argparse

import numpy as np
from matplotlib import pyplot as plt

from Utils.bezier_curve_generator import Bezier

import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

from sklearn.model_selection import train_test_split


def generate_bezier():
    t_points = np.arange(0, 1, 0.01)
    points1 = np.array([[-0.5, 0.6, 0.2], [-0.1, 0.8, -0.4], [0.2, -0.5, -0.3], [0.5, 0.1, 0.1]])
    curve = Bezier.Curve(t_points, points1)
    return curve

def generate_noisy_bezier():
    t_points = np.arange(0, 1, 0.01)
    points1 = np.array([[-0.4, 0.6, 0.1], [0.1, 0.9, -0.2], [0, -0.7, -0.4], [0.5, 0.0, 0.4]])
    # points1+=np.random.uniform(-0.3,0.3,size=(4,3))
    curve = Bezier.Curve(t_points, points1)
    return curve

def generate_cubic_curve():
    x_cub=np.linspace(-0.5, 0.5, 100)
    y_cub=(x_cub*2)**3
    z_cub=(np.sign(x_cub)*np.sqrt(x_cub**2+y_cub**2))**3
    curve=np.concatenate((x_cub, y_cub, z_cub)).reshape((-1, 3), order='F')
    return curve

def get_phi_mat_and_vectors(curve_points):
    U, _, __ = np.linalg.svd(curve_points.T)
    phi=U[:,0:2]
    phi1 = U[:,0]
    phi2 = U[:,1]
    return phi, phi1, phi2

def get_phi1_line(phi1):
    t_points = np.arange(0, 1.2, 0.01)-0.3
    phi1_line=np.array([m*phi1 for m in t_points])
    return phi1_line

def define_network():

    net_input = tf.keras.Input(shape=(1,), dtype=tf.float64)

    net_output = net_input
    net_output = tf.keras.layers.Dense(10, activation='elu', kernel_initializer=HeNormal(), use_bias=False, dtype=tf.float64)(net_output)
    net_output = tf.keras.layers.Dense(10, activation='elu', kernel_initializer=HeNormal(), use_bias=False, dtype=tf.float64)(net_output)
    net_output = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, kernel_initializer=HeNormal(), use_bias=False, dtype=tf.float64)(net_output)

    network = tf.keras.Model(net_input, net_output, name='q_sup_estimator')
    network.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.001), run_eagerly=False, loss='mse')
    network.summary()

    return network

def get_quadratic_q(q_mat):
    ones = np.ones((q_mat.shape[1],q_mat.shape[1]))
    mask = np.array(np.triu(ones,0), dtype=bool, copy=False)

    q_quad_mat=[]
    for q in q_mat:
        out_pr = np.matmul(np.expand_dims(q,axis=1), np.expand_dims(q,axis=0))
        q_quad_mat.append(out_pr[mask])
    q_quad_mat=np.array(q_quad_mat, copy=False)
    return q_quad_mat

def train_quadratic_H(q_train, s_train, phi):
    q_quad = get_quadratic_q(q_train)
    q_quad_inv = np.linalg.pinv(q_quad.T)
    H_mat = (s_train.T - phi@q_train.T)@q_quad_inv
    return H_mat

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('decoder_type', type=str, help='Root directory to work from.')
    args = parser.parse_args()
    decoder_type = args.decoder_type

    curve = generate_cubic_curve()

    phi, phi1, phi2 = get_phi_mat_and_vectors(curve)
    proj_phi1=(phi[:,0:1]@phi[:,0:1].T@curve.T).T

    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 50), np.linspace(-0.3, 0.8, 50))
    d=0.0

    q1_set=(phi[:,0:1].T@curve.T).T
    q2_set=(phi[:,1:2].T@curve.T).T

    if decoder_type=="PODANN":

        normal = np.cross(phi1, phi2)
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        true_projection=(phi@phi.T@curve.T).T
        network = define_network()
        history=network.fit(q1_set, q2_set, epochs=200, batch_size=1)
        approx_q_matrix=np.concatenate((q1_set, network(q1_set).numpy())).reshape((-1, 2), order='F')
        learnt_projection=(phi[:,:]@approx_q_matrix.T).T

    
    elif decoder_type=="Quad":

        H_mat = train_quadratic_H(q1_set, curve, phi[:,0:1])
        H_mat_norm=H_mat/np.linalg.norm(H_mat)

        normal = np.cross(phi1, H_mat_norm.T[0])
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        true_projection=(phi[:,0:1]@q1_set.T+H_mat_norm@H_mat_norm.T@curve.T).T
        learnt_projection=(phi[:,0:1]@q1_set.T+H_mat@get_quadratic_q(q1_set).T).T

    else:
        print('Please, indicate the type of decoder to use: either "PODANN" or "Quad"')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='True manifold')
    ax.plot(proj_phi1[:, 0], proj_phi1[:, 1], proj_phi1[:, 2], 'b-', label='Projection on Phi1 subspace')
    ax.plot_surface(xx, yy, zz, alpha=0.3)
    ax.plot(true_projection[:, 0], true_projection[:, 1], true_projection[:, 2], 'k--', label='Projected manifold')
    ax.plot(learnt_projection[:, 0], learnt_projection[:, 1], learnt_projection[:, 2], 'g-', label='Learnt projection')

    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()