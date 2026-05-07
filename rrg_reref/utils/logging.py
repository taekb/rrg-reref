def rank0_print(message, rank=None):
    '''Print message only if on the main process.'''

    if rank is None or rank == 0:
        print(message)
