min: T;

T > T1;
T > T2;
T > T3;
T > T4;

d1 + d2 + d3 + d4 = V;
d43 + d42 = d4;

tk12 = S12*x12 + C12*d2 + C12*d3 + C12*d4;
tk23 = S23*x23 + C23*d3 + C23*d43;
tk24 = S24*x24 + C24*d42;
tk34 = S34*x34 + C34*d43;

T1 > tk12 + A1*d1;

T2 > tk12 + tk23 + tk24 + A2*d2;
T2 > tk12 + tk23 + tk24 + tk34 - M*y2;

T3 > tk12 + tk23 + tk34 + A3*d3 + tk24 - M*y1;
T3 > tk12 + tk23 + tk34 + A3*d3;
T3 > tk12 + tk23 + tk34 + tk24 - M + M*y2;

T4 > tk12 + tk24 + tk34 + tk23 - M + M*y1 + A4*d4;
T4 > tk12 + tk24 + tk34 - M*y1 + A4*d4;
T4 > tk12 + tk24 + tk34 + tk23 - M*y1 + A4*d43;


x12 < 1;
V*x12 > d2 + d3 + d4;
x23 < 1;
V*x23 > d3 + d43;
x24 < 1;
V*x24 > d42;
x34 < 1;
V*x34 > d43;

y1 < 1;
y2 < 1;
y1 + y2 > 1;

int d1, d2, d3, d42, d43;
int x12, x23, x24, x34;
int y1, y2;
