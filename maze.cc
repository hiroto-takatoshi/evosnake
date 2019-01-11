#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>

using namespace std;

int mat[200][200];

void maze(int gridN = 12) {
    
    assert(gridN % 2 == 0);

    int cnt = 0;
    for (int i = 1; i <= gridN; ++i) {
        mat[1][i] = ++cnt;
    }
    bool fromright = 1;
    for (int i = 2; i < gridN; ++i) {
        if (fromright) {
            for (int j = gridN; j > 1; --j) {
                mat[i][j] = ++cnt;
            }
        }
        else {
            for (int j = 2; j <= gridN; ++j) {
                mat[i][j] = ++cnt;
            }
        }
        fromright = 1 - fromright;
    }
    for (int i = gridN; i >= 1; --i) {
        mat[gridN][i] = ++cnt;
    }

    for (int i = gridN - 1; i > 1; --i) {
        mat[i][1] = ++cnt;
    }

    for (int i = 1; i <= gridN; ++i) {
        for (int j = 1; j <= gridN; ++j) {
            printf("%d,", mat[i][j]);
        }
        printf("\n");
    }

}

int main(int argc, char ** argv) {

    freopen("maze.csv", "w", stdout);

    if (argc == 1) maze();
    else maze(atoi(argv[1]));

    fclose(stdout);

}