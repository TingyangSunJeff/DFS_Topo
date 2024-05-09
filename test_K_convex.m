p = 1e-3:1e-2:1;
zeta = 100;
sigma = 1;
M_1 = 50;
M_2 = 100;
f = (zeta*sqrt(M_1+1) + sigma*sqrt(p)) ./ p + sqrt((M_1+1) * (M_2+1)) ./ p;
plot(p, f);
