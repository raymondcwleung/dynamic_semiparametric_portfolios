library("sgt")

set.seed(7900)
rv = sgt::rsgt(n = 10000, mu = 0, sigma = 1, lambda = -0.25, p = 1.7, q = 7)

start = list(mu = 0, sigma = 1, lambda = 0, p = 2, q = 10)
result = sgt::sgt.mle(X.f = ~ rv, start = start, method = 'nlminb')

q_hi = sgt::qsgt(prob = 0.55, mu = 0, sigma = 1, lambda = 0.0, p = 2.0, q = 1000.0)
