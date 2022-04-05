import multiprocessing

def map(func, iter, ordered=True):
    ncpu = multiprocessing.cpu_count()
    print('You have {0:1d} CPUs'.format(ncpu))
    pool = multiprocessing.Pool(processes=ncpu) 
    inputs = ((func,i) for i in iter) #use a generator, so that nothing is computed before it's needed :)
    try : n = len(iter)
    except TypeError : n = None
    res_list = []
    if ordered: pool_map = pool.imap
    else: pool_map = pool.imap_unordered
   # with tqdm(total=n, desc='# progress ...') as pbar:
    for res in pool_map(_map_f, inputs):
        run = 1
    pool.close()
    pool.join()
    return None