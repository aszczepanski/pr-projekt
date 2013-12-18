min: T;

T > T1;
T > T2;
T > T3;
T > T4;

d1 + d2 + d3 + d4 = V;
d43 + d42 = d4;
d421 + d422 = d42;
d431 + d432 = d43;
d321 + d322 = d3;
d211 + d212 = d2;

tk12 = S12*x12 + C12*d2 + C12*d3 + C12*d4;
tk23 = S23*x23 + C23*d3 + C23*d43;
tk24 = S24*x24 + C24*d42;
tk34 = S34*x34 + C34*d43;

T1 > tk12 + A1*d1;

// y1 = 1, y2 = 0
// drugi model sekwencyjny

T2 > tk12 + tk23 + A2*d2 + tk24 - M*y2;
T2 > tk12 + tk23 + tk34 + A4*d431 + tk24 + A2*d212 - M*y2;

T3 > tk12 + tk23 + tk34 + A3*d3 - M*y2;

T4 > tk12 + tk23 + A2*d211 + tk24 + A4*d432 + A4*d42 - M*y2;
T4 > tk12 + tk23 + tk34 + A4*d4 + tk24 - M*y2;

// y1 = 1, y2 = 1
// trzeci model sekwencyjny

T2 > tk12 + tk23 + tk24 + A2*d2 - M - M + M*y1 + M*y2;

T3 > tk12 + tk23 + tk24 + A4*d421 + tk34 + A3*d322 - M - M + M*y1 + M*y2;
T3 > tk12 + tk23 + A3*d3 + tk34 - M - M + M*y1 + M*y2;

T4 > tk12 + tk23 + tk24 + A4*d4 + tk34 - M - M + M*y1 + M*y2;
T4 > tk12 + tk23 + A3*d321 + tk34 + A4*d422 + A4*d43 - M - M + M*y1 + M*y2;

// y1 = 0, y2 = 1
// pierwszy model sekwencyjny

T2 > tk12 + tk24 + tk23 + A2*d2 - M*y1;

T3 > tk12 + tk24 + A4*d421 + tk34 + A3*d322 - M*y1;
T3 > tk12 + tk24 + tk23 + A3*d3 + tk34 - M*y1;

T4 > tk12 + tk24 + A4*d4 + tk34 - M*y1;
T4 > tk12 + tk24 + tk23 + A3*d321 + tk34 + A4*d422 + A4*d43 - M*y1;

//T2 > tk12 + tk23 + tk24 + A2*d2;
//T2 > tk12 + tk23 + tk24 + tk34 - M*y2;

//T3 > tk12 + tk23 + tk34 + A3*d3 + tk24 - M*y1;
//T3 > tk12 + tk23 + tk34 + A3*d3;
//T3 > tk12 + tk23 + tk34 + tk24 - M + M*y2;

//T4 > tk12 + tk24 + tk34 + tk23 - M + M*y1 + A4*d4;
//T4 > tk12 + tk24 + tk34 - M*y1 + A4*d4;
//T4 > tk12 + tk24 + tk34 + tk23 - M*y1 + A4*d43;


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
//y1 = 0;
//y2 = 1;
int d1, d2, d3, d42, d43, d211, d222, d321, d322, d421, d422, d431, d432;
int x12, x23, x24, x34;
int y1, y2;
