gcc -c -fpic floodfill.c
gcc -shared -o floodfill.so floodfill.o