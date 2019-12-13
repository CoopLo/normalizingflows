import autograd
import autograd.numpy as np
import autograd.scipy as sp
import autograd.misc.optimizers
import numpy
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from autograd.misc.optimizers import adam


def u1(z, N=1):
    exp_factor = 1/2*((np.linalg.norm(z, axis=2) - 2)/0.4)**2 - \
          np.log(np.exp(-1/2*((z[:,:,0] - 2)/0.6)**2) + np.exp(-1/2*((z[:,:,0] + 2)/0.6)**2))
    return N * np.exp(-exp_factor)


def setup_plot():
    X, Y = numpy.mgrid[-4:4:0.05, -4:4:0.05]
    dat = np.dstack((X, Y))
    U_z1 = u1(dat)
    
    fig, ax = plt.subplots()
    ax.contourf(X, Y, U_z1, cmap='Reds', levels=15)
    return ax


def plot_shape():
    ax = setup_plot()
    plt.show()


def plot_shape_samples(samples):
    ax = setup_plot()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=.5)
    plt.show()


# lambda u, w, b
# Flow multiple times
def flow_samples(lambda_flows, z, h):
    D = (lambda_flows.shape[1]-1)//2
    for lambda_flow in lambda_flows:
        z = flow_once(lambda_flow, z, h)
    return z

# Flow once
def flow_once(lambda_flow, z, h):
    D = (lambda_flow.shape[0]-1)//2
    z @ lambda_flow[D:2*D].reshape(-1, 1)
    return z + h((z @ lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1]) @ \
           lambda_flow[:D].reshape(1, -1)

# Psi
def psi(lambda_flow, z, h):
    D = (lambda_flow.shape[0]-1)//2
    return (1-h((z @ lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1])**2) * lambda_flow[D:2*D]


# Calculate energy bound
def energy_bound(lambda_flows, z, h, beta=1.):
    D = (lambda_flows.shape[1]-1)//2
    initial_exp = np.mean(np.log(sp.stats.norm.pdf(z, loc=q_0_mu, scale=np.sqrt(q_0_sigma))))
    joint_exp = beta*np.mean(np.log(u1(flow_samples(lambda_flows, z, h).reshape(1, -1, 2))))
    flow_exp = 0
    for lambda_flow in lambda_flows:
        flow_exp += np.mean(np.log(np.abs(1 + np.dot(psi(lambda_flow, z, h), lambda_flow[:D]))))
        z = flow_once(lambda_flow, z, h)
    return initial_exp - joint_exp - flow_exp


def get_joint_exp(lambda_flows, z, h):
    return np.mean(np.log(u1(flow_samples(lambda_flows, z, h).reshape(1, -1, 2))))


def get_flow_exp(lambda_flows, z, h):
    D = (lambda_flows.shape[1]-1)//2
    flow_exp = 0
    for lambda_flow in lambda_flows:
        flow_exp += np.mean(np.log(np.abs(1 + np.dot(psi(lambda_flow, z, h), lambda_flow[:D]))))
        z = flow_once(lambda_flow, z, h)
        
    return flow_exp


def gradient_descent(m, lambda_flows, grad_energy_bound, samples):
    energy_hist = np.empty(m)
    joint_hist = np.empty(m)
    flow_hist = np.empty(m)
    lambda_hist = np.empty((m, *lambda_flows.shape))
    samples_flowed = samples
    for i in tqdm(range(m)):
        beta = min(1, 0.01+i/10000)
        samples_flowed = flow_samples(lambda_flows, samples, h)
        
        gradient = grad_energy_bound(lambda_flows, samples, h, beta)
        lambda_flows -= step_size*gradient
        #lambda_flows = autograd.misc.optimizers.adam(grad_energy_bound, lambda_flows)
        
        # Debug
        energy_hist[i] = energy_bound(lambda_flows, samples, h)
        joint_hist[i] = get_joint_exp(lambda_flows, samples, h)
        flow_hist[i] = get_flow_exp(lambda_flows, samples, h)
        lambda_hist[i] = lambda_flows
        
        # Plot
        if i % 20 == 0:
            if(i==0):
                leading_zeros = int(np.log(m)/np.log(10))
            elif(i==1000):
                leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10)) - 1
            else:
                leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10))
            zeros = '0'*leading_zeros

            ax = setup_plot()
            ax.scatter(samples_flowed[:, 0], samples_flowed[:, 1], alpha=.5)
            plt.savefig("./plots/{}{}.png".format(zeros, i))
            plt.close()


if __name__ == '__main__':
    #plot_shape()

    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 1
    D = q_0_mu.shape[0]
    num_samples = 1000
    
    num_flows = 5
    lambda_flows = np.array([np.array([1., 0., 4., 5., 0.])]*num_flows)
    
    m = 30000
    step_size = .05
    
    # Samples from initial distribution
    samples = np.random.multivariate_normal(q_0_mu, q_0_sigma*np.eye(D), num_samples)
    #plot_shape_samples(samples)
    
    grad_energy_bound = autograd.grad(energy_bound)

    #gradient_descent(m, lambda_flows, grad_energy_bound, samples)
    #os.system("cd ./plots/ ; convert -delay 10 -loop 0 *.png learning_flows.gif")
    #os.system("cd ./plots/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart learning_flows.mp4")
    def callback(x, i, g):
        left = '['
        right = ']'
        eq = '=' * int(20*i/m)
        blank = ' ' * int(20*(1 - i/m))
        if(i%10 == 0):
            sys.stdout.write("{0}{1}{2}{3}  {4:.3f}%\r".format(left, eq, blank, right, 100*i/m))
            sys.stdout.flush()
        if(i==(m-1)):
            print("[{}]  100%".format(20*'='))

    g_eb = lambda lambda_flows, i: grad_energy_bound(lambda_flows, samples, h)
    output = adam(g_eb, lambda_flows, num_iters=m, callback=callback)
    print(output)
    samples_flowed = flow_samples(output, samples, h)
    ax = setup_plot()
    ax.scatter(samples_flowed[:,0], samples_flowed[:,1], alpha=0.5)
    plt.savefig("./plots/adam_fit.png")
    plt.show()

