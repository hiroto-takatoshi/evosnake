from ctypes import *
import os

body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]

class PyStruct(Structure):
    _fields_ = [
        ("a", c_int),
        ("b", c_int),
        ("c", c_int),
        ("mat", (c_int * 15) * 15),
        ("q", (c_int * 2) * 200),
        ("fx", c_int),
        ("fy", c_int),
    ]

so = CDLL(os.path.abspath("floodfill.so"))
so.floodfill.argtypes = [POINTER(PyStruct)]
so.floodfill.restype = None

print(so)

ps = PyStruct()

for x in body[1:]:
    ps.mat[x[0]][x[1]] = -1
ps.q[1][0] = body[0][0]
ps.q[1][1] = body[0][1]
ps.fx = 3
ps.fy = 3

so.floodfill(byref(ps))

print(ps.a, ps.b, ps.c)