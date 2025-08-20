# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd


#plotting libraries
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

cmap = plt.cm.viridis
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=2),
            Line2D([0], [0], color=cmap(.25), lw=2),
            Line2D([0], [0], color=cmap(.50), lw=2),
            Line2D([0], [0], color=cmap(.75), lw=2),
            Line2D([0], [0], color=cmap(1.), lw=2)]



logger = logging.getLogger('wot')


def compute_transport_matrix(solver, **params):
    """
    Compute the optimal transport with stabilized numerics.
    Args:
    G: Growth (absolute)
    solver: transport_stablev2 or optimal_transport_duality_gap
    growth_iters:
  """

    import gc
    G = params['G']
    pri_df_final = pd.DataFrame()
    growth_iters = params['growth_iters']
    learned_growth = []
    for i in range(growth_iters):
        if i == 0:
            row_sums = G
        else:
            row_sums = tmap.sum(axis=1)  # / tmap.shape[1]
        params['G'] = row_sums
        learned_growth.append(row_sums)
        tmap,pri_df = solver(**params) #Added a dataframe with values of primal to the output of the solver
        pri_df_final = pd.concat([pri_df_final,pri_df],axis=1)

        fig,axs= plt.subplots(2,3,sharex=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.suptitle(f'Loss function for {i} iterate')
        plt.xticks(fontsize=12)
        #plt.figure(figsize=(20, 20))
        #ax.size(20,20)
        #axs[0].plot(pri_df.T['P1'],pri_df.T['P2'],pri_df.T['P3'],pri_df.T['P4'],pri_df.T['dg'])
        pri_df.T.plot(ax=axs[0,0],colormap=cmap)
        axs[0,0].set_xlabel("Interation",fontsize=18)
        axs[0,0].legend(custom_lines, ['1st term', '2nd Term', '3rd term','4th term','Duality Gap'],
                      loc='upper right',prop={'size':14})
        axs[0,1].plot(pri_df.T['P1'],'--k')
        axs[0,1].title.set_text('First Term')
        
        axs[0,2].plot(pri_df.T['P2'],'--k')
        axs[0,2].title.set_text('Second Term')
        
        axs[1,0].plot(pri_df.T['P3'],'--k')
        axs[1,0].title.set_text('Third Term')
        
        axs[1,1].plot(pri_df.T['P4'],'--k')
        axs[1,1].title.set_text('Fourth Term')
        
        axs[1,2].plot(pri_df.T['dg'],'--k')
        axs[1,2].title.set_text('Duality Gap')

        gc.collect()

    return tmap,pri_df_final, learned_growth #edited to output pri_df


# @ Lénaïc Chizat 2015 - optimal transport
def fdiv(l, x, p, dx):
    return np.sum(l * (dx * (x * (np.log(x / p)) - x + p)))

# def new_fdiv(l, x, p, dx):
#     return (l * (dx * (x * (np.log(x / p)) - x + p)))


def fdivstar(l, u, p, dx):
    return np.sum(l*(p * dx) * (np.exp(u / l) - 1))

def primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(np.dot(R, dy), dx) + F2(np.dot(R.T, dx), dy) \
               + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) \
                  + np.sum(R * C)) / (I * J)

def primal_first_term(C,K,R,p,q,epsilon):
    I = len(p)
    J = len(q)
    with np.errstate(divide='ignore'):
        return np.sum(R * C) / (I * J)


def primal_second_term(C,K,R,p,q,epsilon):
    I = len(p)
    J = len(q)
    with np.errstate(divide='ignore'):
        return (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K)) / (I * J)


def primal_third_term(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(np.dot(R, dy), dx)


def primal_ind_third_term(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    with np.errstate(divide='ignore'):
        return np.sum(abs(np.dot(R, dy) - p))


def primal_fourth_term(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F2(np.dot(R.T, dx), dy) 


# def primal_ind_fourth_term(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
#     I = len(p)
#     J = len(q)
#     F1 = lambda x, y: fdiv(lambda1, x, p, y)
#     F2 = lambda x, y: new_fdiv(lambda2, x, q, y)
#     with np.errstate(divide='ignore'):
#         return F2(np.dot(R.T, dx), dy) 



def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    return - F1c(- epsilon * np.log(a), dx) - F2c(- epsilon * np.log(b), dy) \
           - epsilon * np.sum(R - K) / (I * J)




# Add another method for calculating partially balanced optimal transport

## Helper functions 

#Review changes made the fdivstar
def new_fdiv(l,x, p,dx,delta):
    if max(abs(x-p)) < delta:
    # if np.sum(abs(x - p)) < 1e-10:
    # if np.array_equal(x,p):
        return 0.0
    else:
        return 10000

def new_fdivstar(l, u, p, dx, delta):
    if max(abs(u)) <= 10000 / delta:
        return np.sum(p * dx * u)
    else:
        return 10000


#changes to enforce colsum
# Change the calculatiof of dual and primal for the penalty to be an indicator function
def new_primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2,delta):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1,x,p,y)  # noqa: E731
    F2 = lambda x, y: new_fdiv(lambda2,x,q,y,delta)
    return F1(np.dot(R,dy),dx) + F2(np.dot(R.T,dx),dy) \
            + (np.sum(epsilon * R * np.nan_to_num(np.log(R)) - epsilon * R + epsilon * K) \
            + np.sum(R * C)) / (I * J)



def new_dual(C, K, R, dx, dy, p, q, a, b,epsilon, lambda1, lambda2, delta):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: new_fdivstar(lambda2, u, q,v,delta)
    return -F2c( -epsilon * np.log(b), dy) - F1c( -epsilon * np.log(a),dx) \
            - np.sum(epsilon*(R-K)) / (I * J)   


# ## changes to enforce rowsum
# # Change the calculatiof of dual and primal for the penalty to be an indicator function
# def new_primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2,delta):
#     I = len(p)
#     J = len(q)
#     F1 = lambda x, y: new_fdiv(lambda1,x,p,y,delta)
#     F2 = lambda x, y: fdiv(lambda2,x,q,y)
#     return F1(np.dot(R,dy),dx) + F2(np.dot(R.T,dx),dy) \
#             + (np.sum(epsilon * R * np.nan_to_num(np.log(R)) - epsilon * R + epsilon * K) \
#             + np.sum(R * C)) / (I * J)



# def new_dual(C, K, R, dx, dy, p, q, a, b,epsilon, lambda1, lambda2, delta):
#     I = len(p)
#     J = len(q)
#     F1c = lambda u, v: new_fdivstar(lambda1, u, p, v,delta)
#     F2c = lambda u, v: fdivstar(lambda2, u, q,v)
#     return -F2c( -epsilon * np.log(b), dy) - F1c( -epsilon * np.log(a),dx) \
#             - np.sum(epsilon*(R-K)) / (I * J)   





# end @ Lénaïc Chizat
'''
def rank_penalty(df,rank_column='rank_column',lambda3 = 1):
    """
    Addes a penatly to the cost matrix for transitions which are going in the 'wrong' direction

    """
    if(
    return beta
'''




def optimal_transport_duality_gap(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter, **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`

    Parameters
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.

    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J

    p = G
    # q = np.ones(C.shape[1]) * np.average(G)
    #change q to be all 1s
    q = np.ones(C.shape[1])
    print("this is the value of q",q)

    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0
    pri_df = pd.DataFrame()

    

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6
        

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)

                if current_iter >= max_iter:
                    logger.warning("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                # Change primal and dual to be from new functions
                pri = primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
                #calculate terms of the primal function separately
                pri_first = primal_first_term(C, _K, R,p, q, epsilon_i)
                pri_second = primal_second_term(C, _K, R, p, q, epsilon_i)
                pri_third = primal_third_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                pri_fourth = primal_fourth_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                #save output to dataframe for plotting later
                primal_dictionary = {'P1':pri_first,'P2':pri_second,'P3':pri_third,'P4':pri_fourth,'dg':duality_gap}
                int_df = pd.DataFrame.from_dict(primal_dictionary,orient='index')
                pri_df = pd.concat([pri_df,int_df],axis=1)
                #print('This is the duality gap',duality_gap)
            else:
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))
                #print("this is duality gap when epsilon is high",duality_gap)

    pri_df.columns = range(pri_df.shape[1])
    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1],pri_df






def optimal_transport_partial_balanced(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter,delta, **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`

    Parameters
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.

    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)
    
    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J
    
    p = G
    q = np.ones(C.shape[1]) * (J/I)
    
    
    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)
    
    epsilon_i = epsilon0 * scale_factor
    current_iter = 0
    pri_df = pd.DataFrame()
    
    
    # print("Lets see if its working")
    
    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        #_K = np.exp((-C * 
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6
        
    
        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                b = (q / (K.T.dot(np.multiply(a, dx))))
                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)
                if current_iter >= max_iter:
                    logger.warning("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b    
    
            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)
    
            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                a = np.array(a)
                R = (K.T * a).T * b
                pri = new_primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2,delta)
                dua = new_dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2,delta)
                duality_gap = abs(pri - dua) / abs(pri)
                #calculate individual loss terms
                pri_first = primal_first_term(C, _K, R,p, q, epsilon_i)
                pri_second = primal_second_term(C, _K, R, p, q, epsilon_i)
                pri_third = primal_ind_third_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                pri_fourth = primal_fourth_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                #save output to dataframe for plotting later
                primal_dictionary = {'P1':pri_first,'P2':pri_second,'P3':pri_third,'P4':pri_fourth,'dg':duality_gap}
                int_df = pd.DataFrame.from_dict(primal_dictionary,orient='index')
                pri_df = pd.concat([pri_df,int_df],axis=1)
            else:
                a = np.array(a)
                R = (K.T * a).T * b
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))
            
    pri_df.columns = range(pri_df.shape[1])
    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1],pri_df




# keep this for later
def optimal_transport_partial_balanced_rowsum(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter,delta, **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`

    Parameters
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.

    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)
    
    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J
    
    p = G
    q = np.ones(C.shape[1]) * np.average(G)
    
    
    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)
    
    epsilon_i = epsilon0 * scale_factor
    current_iter = 0
    pri_df = pd.DataFrame()
    
    
    # print("Lets see if its working")
    
    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        #_K = np.exp((-C * 
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6
        
    
        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy))))
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))
                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)
                if current_iter >= max_iter:
                    logger.warning("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b    
    
            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)
    
            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                a = np.array(a)
                R = (K.T * a).T * b
                pri = new_primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2,delta)
                dua = new_dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2,delta)
                duality_gap = abs(pri - dua) / abs(pri)
                #calculate individual loss terms
                pri_first = primal_first_term(C, _K, R,p, q, epsilon_i)
                pri_second = primal_second_term(C, _K, R, p, q, epsilon_i)
                pri_third = primal_ind_third_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                pri_fourth = primal_fourth_term(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                #save output to dataframe for plotting later
                primal_dictionary = {'P1':pri_first,'P2':pri_second,'P3':pri_third,'P4':pri_fourth,'dg':duality_gap}
                int_df = pd.DataFrame.from_dict(primal_dictionary,orient='index')
                pri_df = pd.concat([pri_df,int_df],axis=1)
            else:
                a = np.array(a)
                R = (K.T * a).T * b
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))
            
    pri_df.columns = range(pri_df.shape[1])
    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1],pri_df





