from joblib import Parallel, delayed
r = Parallel(n_jobs=-1)( [delayed(process)(i) for i in range(10000)])
