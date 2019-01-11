#include <stdio.h>
#include <memory.h>
#include <math.h>

typedef struct TestStruct {
    int     a, b, c;
    int   mat[15][15];
    int     q[200][2];
    int     fx, fy;
    float   d, e;
} TestStruct;

extern void floodfill(TestStruct *ts) {
    int i, j, h = 1, t = 1;
    int d[][2] = {0, 1, 0, -1, 1, 0, -1, 0};
    ts -> a = 0;
    while (h <= t) {
        int ux = ts -> q[h][0], uy = ts -> q[h][1];
        ++h;
        //printf("!");
        for (i = 0; i < 4; ++i) {
            int vx = ux + d[i][0], vy = uy + d[i][1];
            if (ts -> mat[vx][vy] == -1 || vx <= 0 || vy <= 0 || vx >= 13 || vy >= 13) continue;
            if (ts -> mat[vx][vy] > 0) continue;
            int x = ts -> mat[ux][uy] + 1;
            ts -> mat[vx][vy] = x;
            if (x > ts -> a) {
                ts -> a = x;
            }
            ts -> q[++t][0] = vx;
            ts -> q[t][1] = vy;
        }
    }
    int x = ts -> mat[ts -> fx][ts -> fy];
    //if (x <= 0) x = 200;
    ts -> b = x;
    ts -> c = t;
    int xmin = 20, xmax = 0, ymin = 20, ymax = 0;
    float tot = 0, tot1 = 0.0, tot2 = 0.0;
    for (i = 1; i <= 12; ++i)
        for (j = 1; j <= 12; ++j) {
            if (ts -> mat[i][j] == -1) {
                xmin = i < xmin ? i : xmin;
                xmax = i > xmax ? i : xmax;
                ymin = j < ymin ? j : ymin;
                ymax = i > ymax ? j : ymax;

                tot += (float)sqrt( (i - ts -> fx) * (i - ts -> fx) + (j - ts -> fy) * (j - ts -> fy) );
                tot1 += 1;
            }
        }
    
    ts -> d = tot1 / ((xmax - xmin + 1) * (ymax - ymin + 1));
    ts -> e = tot / (tot1 + 1);
}