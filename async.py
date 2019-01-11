from multiprocessing import Pool
import time

def f(x):
    for i in range(1000000):
        j = i + 1
        j -= 1

if __name__ == '__main__':
    t = time.time()
    for i in range(10):
        f(4)
    print(time.time() - t)

    t = time.time()
    with Pool(10) as p:
        p.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    print(time.time() - t)