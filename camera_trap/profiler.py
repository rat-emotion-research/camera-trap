from cProfile import Profile
from pstats import Stats

def profiler(logfile):
    def decorator(foo):
        def wrapper(*args, **kwargs):
            pr = Profile()
            pr.runcall(foo, *args, **kwargs)
            pr.dump_stats(logfile)

        return wrapper
    return decorator
    