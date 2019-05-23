import matplotlib.pyplot as plt
import numpy as np



N = 4 # number of players

K = 100 # number of actions

BS = np.arange(K) / K # bids discretized

P = 100 # number of policies

# coefficients for policies pi[i](x) = floor(c[i]*x*K)/K
C = np.arange(1, P+1) / P
U = np.array([1./K] * K)



def do_policy(pi, x):
    # pi scalar (int)
    # x vector
    return (C[pi]*x*K).astype(int)

def do_policy_scalar(pi, x):
    # pi scalar (int)
    # x scalar
    return int(C[pi]*x*K)

def sample(x, Q, pi_default, mu):
    # x scalar, TODO: do with x as a numpy array
    # this seems annoying because not only
    # do we have to work with 2d indices/arrays in np.add.at
    # we also have to bucket x into groups, because
    # otherwise, this is too memory intensive for large x
    # since we have 2D arrays of size K*len(x) which
    # is on the order of 1GB for the constants we're working with

    Q = np.copy(Q)
    Q[pi_default] += 1 - Q.sum()

    #print('Q=', Q)

    p = np.zeros(K)
    np.add.at(p, (C*x*K).astype(int), Q)
    #     for i in range(P):
    #         p[int(C[i]*x*K)] += Q[i]
    p = (1 - K*mu) * p + mu

    #print(p)

    a = np.random.multinomial(1, p).argmax()

    return a, p[a]



def ips_reward(pi, x, a, r, p):
    # pi scalar (int)
    #x, a, r, p vectors

    acs = do_policy(pi, x)
    return np.mean((r/p) * (acs == a))

def compute_mu(tau, delta, mu_coeff):
    return min(0.5/K, mu_coeff*(np.log(16*tau**2*P/delta) / (K*tau))**0.5)


def second_price_auction(A, X):
    assert A.shape == X.shape
    N, tau = A.shape
    R = np.zeros((N, tau))

    AT = A.T.copy()
    AT.sort()

    for i in range(N):
        R[i] = (A[i] == AT[:,-1]) * (X[i] - BS[AT[:,-2]])

    return R


def first_price_auction(A, X):
    assert A.shape == X.shape
    N, tau = A.shape
    R = np.zeros((N, tau))

    Am = A.max(axis=0)

    for i in range(N):
        R[i] = ((A[i] == Am)
                * (X[i] - BS[A[i]])) # 1st price auction

    return R

def Q_mu(Q, mu, a, x):
    # Q vector
    # mu, a, x scalars

    pr_a = Q[(C*x*K).astype(int) == a].sum()

    return (1 - K*mu)*pr_a + mu



def update_Q(X, A, R, Pr, mu, Qinit):

    # Coordinate descent algorithm

    ipss = np.array([ips_reward(pi, X, A, R, Pr) for pi in range(P)])

    # in general this is an oracle
    pi_best = ipss.argmax()

    psi = 100.0
    b = np.array([(ipss[pi_best] - ipss[i])/(psi*mu) for i in range(P)])

    Q = np.copy(Qinit)

    while True: #for _ in range(100):
        qkb_sum = (Q * (2*K + b)).sum()
        if qkb_sum > 2*K:
            Q *= 2*K / qkb_sum

        V = np.array([np.mean([1 / Q_mu(Q, mu, do_policy_scalar(pi, x), x)
                               for x in X
                              ])
                      for pi in range(P)])
        D = V - (2*K + b)

        pi = D.argmax()
        if D[pi] > 0:
            S_pi = np.mean([1 / Q_mu(Q, mu, do_policy_scalar(pi, x), x)**2
                            for x in X
                            ])

            a_pi = (V[pi] + D[pi]) / (2 * (1 - K*mu) * S_pi)

            Q[pi] += a_pi
        else:
            break

    return Q, pi_best


def bandit_dynamics(epochs=10, mu_coeff=1.0, plot=False):

    c_best_hist = []

    taus = [0] + [2**k for k in range(epochs)]
    delta = 0.1 # probability of failure

    t = 0

    pi_best = [np.random.randint(0,P) for i in range(N)]
    c_best_hist = [C[pi_best]]

    Hs = {i : ([], [], [], []) for i in range(N)}
    #tuple of ((xt, at, rt(at), pt(at)))

    Qs = {i : np.zeros(P) for i in range(N)}

    tot_welf = 0.0
    tot_tau = 0

    welf_hist = []


    for tau in taus[1:]:
        mu = compute_mu(tau, delta, mu_coeff)
        t += tau

        Pr = np.zeros((N, tau)) # probabilities
        A = np.zeros((N, tau), dtype=int) # actions, i.e. bids

        X = np.random.random((N, tau))

        for i in range(N):
            a, p = zip(*[sample(x, Qs[i], pi_best[i], mu) for x in X[i]])
            A[i] = np.array(a)
            Pr[i] = np.array(p)

        R = first_price_auction(A, X)

        for i in range(N):
            x, a, r, p = Hs[i]
            Hs[i] = (np.concatenate((x, X[i])),
                     np.concatenate((a, A[i])),
                     np.concatenate((r, R[i])),
                     np.concatenate((p, Pr[i])))

        for i in range(N):
            Qs[i], pi_best[i] = update_Q(*Hs[i], mu, Qs[i])

        c_best_hist.append(C[pi_best])



        Rsum = R.sum()
        tot_tau += tau
        tot_welf += Rsum
        welf_hist.append(tot_welf / tot_tau)

        #print(tot_tau, C[pi_best])

    #cs_hist = zip(*c_best_hist)
    #for cs in cs_hist:
    #    plt.plot(taus, cs)

    welf_hist = np.array(welf_hist)

    if plot:
        plt.plot(taus[1:], welf_hist / 0.2)



        plt.xscale('log')
        plt.xlabel('total number of samples')
        plt.ylabel('Fraction of optimal welfare')
        plt.title('First price auction dynamics welfare')


    return welf_hist

if __name__ == '__main__':
    bandit_dynamics()
