from concurrent.futures import ThreadPoolExecutor

class Thread_Pool():

    def __init__(self, max_thread_num):
        self.max_thread_num = max_thread_num

    def run(self, function, args):
        with ThreadPoolExecutor(self.max_thread_num) as executor:
            executor.map(function, args)

def run_thread_pool(*dargs, **dkargs):
    def wrapper(func):
        def inner(*args):
            thread_pool = Thread_Pool(*dargs)
            thread_pool.run(func, *args)
        return inner
    return wrapper



if __name__ == '__main__':
    pass