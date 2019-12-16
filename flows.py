import autograd
import autograd.numpy as np
import autograd.scipy as sp
import autograd.misc.optimizers
import numpy
import time
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from autograd.misc.optimizers import adam

e_bound = []
joint_probs = []
flow_probs = []
grad_norms = []
def callback(x, i, g):
    left = '['
    right = ']'
    eq = '=' * int(20*i/m)
    blank = ' ' * int(20*(1 - i/m))
    grad_norms.append(np.linalg.norm(g))
    if(i%10 == 0):
        sys.stdout.write("{0}{1}{2}{3}  {4:.3f}%  {5:.2f}s\r".format(
                         left, eq, blank, right, 100*i/m, time.time()-start))
        sys.stdout.flush()
    if(i==(m-1)):
        sys.stdout.write("{}\r".format(' '*50))
        sys.stdout.flush()
        print("[{}]  100%  {}".format(20*'=', time.time() - start))
    if(i%100 == 0):
        new_samples = np.random.randn(num_samples)[:,np.newaxis]
        flowed_samples = flow_samples(x, new_samples, np.tanh)
        fig, ax = plt.subplots()
        ax.hist(flowed_samples, bins=35, density=True)
        plt.savefig("./data_fit_1d/{}.png".format(i))
        plt.close()


def w1(z):
    return np.sin(2*np.pi*z/4)

def w2(z):
    return 3*np.exp(-1/2*((z-1)/0.6)**2)

def u1(z, N=1):
    exp_factor = 1/2*((np.linalg.norm(z, axis=2) - 2)/0.4)**2 - \
          np.log(np.exp(-1/2*((z[:,:,0] - 2)/0.6)**2) + np.exp(-1/2*((z[:,:,0] + 2)/0.6)**2))
    return N * np.exp(-exp_factor)


def u2(z, N=1):
    exp_factor = 1/2*((z[:,:,1] - np.sin(2*np.pi*z[:,:,0]/4))/0.4)**2
    return np.exp(-exp_factor)


def u3(z, N=1):
    exp_factor = -np.log(np.exp(-1/2*((z[:,:,1] - w1(z[:,:,0]))/0.35)**2) + \
                  np.exp(-1/2*((z[:,:,1] - w1(z[:,:,0]) + w2(z[:,:,0]))/0.35)**2))
    return np.exp(-exp_factor)


def setup_plot(u_func):
    try:
        X, Y = numpy.mgrid[-4:4:0.05, -4:4:0.05]
        dat = np.dstack((X, Y))
        U_z1 = u_func(dat)
        
        fig, ax = plt.subplots()
        ax.contourf(X, Y, U_z1, cmap='Reds', levels=15)
    except (TypeError, ValueError):
        plt.close()
        x = np.linspace(-8, 8, 1000)
        fig, ax = plt.subplots()
        ax.plot(x, u_func(x))
    return ax


def plot_shape():
    ax = setup_plot(u_func)
    plt.show()


def plot_shape_samples(samples, u_func):
    ax = setup_plot(u_func)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=.5)
    plt.show()


# Flow once
def flow_once(lambda_flow, z, h):
    D = (lambda_flow.shape[0]-1)//2
    return z + h((z @ lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1]) @ \
           lambda_flow[:D].reshape(1, -1)


# lambda u, w, b
# Flow multiple times
def flow_samples(lambda_flows, z, h):
    D = (lambda_flows.shape[1]-1)//2
    for lambda_flow in lambda_flows:
        z = flow_once(lambda_flow, z, h)
    return z

# Psi
def psi(lambda_flow, z, h):
    D = (lambda_flow.shape[0]-1)//2
    return (1-h((z @ lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1])**2) * lambda_flow[D:2*D]


# Calculate energy bound
def energy_bound(lambda_flows, z, h, u_func, beta=1.):
    D = (lambda_flows.shape[1]-1)//2
    #initial_exp = np.mean(np.log(sp.stats.norm.pdf(z, loc=q_0_mu, scale=np.sqrt(q_0_sigma))))
    initial_exp = 0
    joint_exp = beta*np.mean(np.log(u_func(flow_samples(lambda_flows, z, h).reshape(1, -1, 2))))

    flow_exp = 0
    for k, lambda_flow in enumerate(lambda_flows):
        flow_exp += np.mean(np.log(np.abs(1 + np.dot(psi(lambda_flow, z, h), lambda_flow[:D]))))
        z = flow_once(lambda_flow, z, h)

    e_bound.append((initial_exp - joint_exp - flow_exp)._value)
    joint_probs.append(joint_exp._value)
    #flow_probs.append(flow_exp._value)
    flow_probs.append(0)
    return initial_exp - joint_exp - flow_exp


def get_joint_exp(lambda_flows, z, h, u_func):
    return np.mean(np.log(u_func(flow_samples(lambda_flows, z, h).reshape(1, -1, 2))))


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

            ax = setup_plot(u_func)
            ax.scatter(samples_flowed[:, 0], samples_flowed[:, 1], alpha=.5)
            plt.savefig("./plots/{}{}.png".format(zeros, i))
            plt.close()


def adam_solve(lambda_flows, grad_energy_bound, samples, u_func, h, m=1000, step_size=0.001):
    output = np.copy(lambda_flows)
    print("BEFORE LEARNING:\n{}".format(output))
    grad_energy_bound = autograd.grad(energy_bound)
    g_eb = lambda lambda_flows, i: grad_energy_bound(lambda_flows, samples, h, u_func, 
                                                     #beta= (0.1 + i/1000))
                                                     beta=min(1, 0.01+i/10000))
    output = adam(g_eb, output, num_iters=m, callback=callback, step_size=step_size)
    print("AFTER LEARNING:\n{}".format(output))
    samples_flowed = flow_samples(output, samples, h)
    np.savetxt("./data_fit_1d/flow_params.txt", output)
    return samples_flowed


def shape_fit_2d(m, step_size, u_func, num_flows=8, num_samples=1000):
    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 5
    D = q_0_mu.shape[0]
    num_samples = 2000
    
    num_flows = 15

    # 2D flows
    #lambda_flows = np.array([np.array([1., 0., 4., 5., 0.])]*num_flows)
    lambda_flows = np.array([np.array([1., 1., 0., 0., 0.])]*num_flows)

    # 2D samples
    samples = np.random.multivariate_normal(q_0_mu, q_0_sigma*np.eye(D), num_samples)

    # Gradient of energy function -> used to minimize energy
    grad_energy_bound = autograd.grad(energy_bound)

    #gradient_descent(m, lambda_flows, grad_energy_bound, samples)
    flowed_samples = adam_solve(lambda_flows, grad_energy_bound, samples,
                                u_func, h, m, step_size)

    # Plot Transformed samples
    ax = setup_plot(u_func)
    print(samples_flowed.shape)
    ax.scatter(samples_flowed[:,0], samples_flowed[:,1], alpha=0.2)
    plt.savefig("./plots/adam_fit_test.png")

    # Convert plots to gif or mp4
    #os.system("cd ./plots/ ; convert -delay 10 -loop 0 *.png learning_flows.gif")
    #plot_str = "cd ./plots/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v "
    #plot_str += "libx264 -pix_fmt yuv420p -movflags +faststart learning_flows.mp4"
    #os.system(plot_str)


def shape_fit_1d(m, step_size, u_func, num_flows=8, num_samples=1000):
    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 5
    D = q_0_mu.shape[0]

    # 1D flows
    lambda_flows = np.array([np.array([1., 1., 0.])]*num_flows)

    # 1D samples
    samples = np.random.randn(num_samples)[:,np.newaxis]
    
    start = time.time()
    grad_energy_bound = autograd.grad(energy_bound)

    # JOINT PROBABILITY IS NEW U_FUNC
    #print(energy_bound(lambda_flows, samples, h, u_func))

    #target = lambda x: (sp.stats.norm.pdf((x-2)) + sp.stats.norm.pdf((x+2)))/2

    #gradient_descent(m, lambda_flows, grad_energy_bound, samples)
    flowed_samples = adam_solve(lambda_flows, grad_energy_bound, samples,
                                u_func, h, m, step_size)

    # Plot Transformed samples
    ax = setup_plot(u_func)
    ax.hist(flowed_samples, bins=50, alpha=0.5, density=True)
    plt.savefig("./plots/adam_fit_test.png")

    # Convert plots to gif or mp4
    #os.system("cd ./plots/ ; convert -delay 10 -loop 0 *.png learning_flows.gif")
    #plot_str = "cd ./plots/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v "
    #plot_str += "libx264 -pix_fmt yuv420p -movflags +faststart learning_flows.mp4"
    #os.system(plot_str)


def data_fit_1d(x_dat, m, step_size, u_func, num_flows=8, num_samples=1000):
    # Set target data
    ufunc = lambda z: u_func(x_dat, z)

    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 5
    D = q_0_mu.shape[0]

    # 1D flows
    lambda_flows = np.array([np.array([1., 1., 0.])]*num_flows)

    # 1D samples
    samples = np.random.randn(num_samples)[:,np.newaxis]
    
    # Gradient of energy bound
    grad_energy_bound = autograd.grad(energy_bound)

    # Minimize with adam, also time it
    start = time.time()
    flowed_samples = adam_solve(lambda_flows, grad_energy_bound, samples,
                                ufunc, h, m, step_size)

    # Plot Transformed samples
    fig, ax = plt.subplots()
    ax.hist(x_dat, bins=50, alpha=0.5, density=True, label="Target")
    ax.hist(flowed_samples, bins=50, alpha=0.5, density=True, label="Flowed Samples")
    ax.legend(loc='best')
    plt.savefig("./plots/data_fit_1d.png")
    #plt.show()


def sample(u_func, shape, llim, ulim, num_samples=1000):
    # Sample target distribution with rejection sampling
    samples = []
    while(len(samples) < num_samples):
        trial_sample = np.random.uniform(llim, ulim, size=shape)
        if(u_func(trial_sample) > np.random.random(shape)):
            samples.append(trial_sample[0])
    return np.array(samples)[:,np.newaxis]


def u_func(x, z):
    z = z[0].reshape((-1, 1))
    return (sp.stats.norm.pdf(x-5) + sp.stats.norm.pdf(x-5))/2 * \
           (sp.stats.norm.pdf(x-z))


if __name__ == '__main__':
    m = 20000
    #u_func = u1   # 2D Shape fit
    u_func = lambda x: (sp.stats.norm.pdf((x-4)) + sp.stats.norm.pdf((x+4)))/2  # 1D Shape fit

    target = lambda x: (sp.stats.norm.pdf((x-5)) + sp.stats.norm.pdf((x+5)))/2  # 1D Shape fit
    #u_func = lambda x, z: (sp.stats.norm.pdf(x-4-z) + sp.stats.norm.pdf(x-4-z))/2 * \
    #                      (sp.stats.norm.pdf(x-4) + sp.stats.norm.pdf(x-4))/2
    print("SAMPLING")

    #TODO: Be able to use different number of target and initial samples
    num_samples = 1000
    x_dat = sample(target, 1, -8, 8, num_samples)

    #plt.show()
    step_size = .001
    num_flows = 5
    start = time.time()
    #shape_fit_2d(m, step_size, u_func, num_flows, num_samples)
    shape_fit_1d(m, step_size, u_func, num_flows, num_samples)
    #data_fit_1d(x_dat, m, step_size, u_func, num_flows, num_samples)
    
    fig, ax = plt.subplots(nrows=4, figsize=(8,20))
    ax[0].plot(grad_norms, label="Norm of gradient")
    ax[0].legend(loc='best')
    ax[1].plot(e_bound, label="Energy Bound")
    ax[1].legend(loc='best')
    ax[2].plot(joint_probs, label="Joint Probability")
    ax[2].legend(loc='best')
    ax[3].plot(flow_probs, label="Flow Probability")
    ax[3].legend(loc='best')
    plt.show()

