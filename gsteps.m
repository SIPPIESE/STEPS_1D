signal = readmatrix("sample_59570Hz_vib_time.txt", NumHeaderLines = 5);
fftsig = abs(fftshift(fft(signal(:,2))));
halfsig = fftsig(round(size(fftsig, 1)/2) + 2: end);
% plot(halfsig)
str1 = "39040, 78013, 116989, 117053, 117120, 117254, 234509, 273549";
arg = str2num(str1);
[out1, out2, out3] = gstep(signal(:, 2).', arg);


function [f_est, A_est, phi_est] = gstep(x0, k0)
N = length(x0);
% set parameters of solvers
options = optimoptions('fsolve');
options.Algorithm = 'levenberg-marquardt';
options.Display = 'off';
precision = 1e-5;
options.FunctionTolerance = precision;
options.TolX = precision;
options.StepTolerance = precision;
% get initial values by ipDFT
n = 0:1:N-1;
x = x0.*hann(N, "periodic")';
M = length(k0);
X1 = zeros(M, 1);
X_est_0 = x*exp(-1j*2*pi*n'*k0/N);
X1 = X_est_0;
X_est_l = x*exp(-1j*2*pi*n'*(k0-1)/N);
X_est_u = x*exp(-1j*2*pi*n'*(k0+1)/N);
Mag0 = abs(X_est_0);
Magl = abs(X_est_l);
Magu = abs(X_est_u);
delta = 2*(Magu-Magl)./(Magu+2*Mag0+Magl);
k_ini = k0 + delta;
W_d = abs(2*sin(pi*delta)./sin(pi*delta/N).*exp(1j*pi*(N-1)/N*delta)-...
    sin(pi*(1-delta))./sin(pi*(1-delta)/N).*exp(-1j*pi*(N-1)/N*(1-delta))-...
    sin(pi*(1+delta))./sin(pi*(1+delta)/N).*exp(1j*pi*(N-1)/N*(1+delta)));
index = delta==0;
W_d(index) = abs(2*pi/(pi/N))*ones(length(find(index==1)));
A_ini = 8*abs(X_est_0)./(W_d);
row_ones = ones(1,M);
col_ones = ones(M,1);
k_ini_mat = k_ini'*row_ones;
% initial frequency
k_ini1 = k_ini(1);
if mod(k_ini1, 2)~=0
    coef2 = ((floor(k_ini1)+1)/2)/(k_ini1);
else
    coef2 = (floor(k_ini1)/2)/(k_ini1);
end
% coef2 = floor(k_ini1)/k_ini1;
D1 = 1;
D2 = round(N*coef2)/N;
N1 = round(D1*N);
N2 = round(D2*N);
if mod(N2,2)~=0
    N2=N2-1;
    D2=N2/N;
end
ns1 = -(N1-1)/2:1:(N1-1)/2;
x1 = x0(1: N1);
X1 = x1*exp(-1j*2*pi/N1*ns1'*D1*k_ini);
phi_ini = angle(X1);
X1 = transpose(X1);
ns2 = -(N2-1)/2:1:(N2-1)/2;
x2 = x0(round(N*(1-D2))+1: N);
X2 = x2*exp(-1j*2*pi/N2*ns2'*D2*k_ini);
X2 = transpose(X2);
t0 = [k_ini;A_ini;phi_ini];
% solve nonlinear equations
fun = @(t)nlEqu(t);
[t_hat, ~] = fsolve(fun, t0, options);
f_est = t_hat(1, :);
A_est = t_hat(2, :);
phi_est = t_hat(3, :);

% if phi_est-phi0 > pi
%     phi_est = phi_est-2*pi;
% elseif phi_est-phi0 < -pi
%     phi_est = phi_est+2*pi;
% end
    % objective function
    function F = nlEqu(t)
            f_mat = col_ones*t(1, :);
            phase1 = col_ones*t(3, :);
            phase2 = phase1+pi*f_mat*(1-D2);
            A = t(2, :)';
            % equations
            a1 = sin(D1*pi*(k_ini_mat-f_mat))./sin(pi*(k_ini_mat-f_mat)/N);
            a1(k_ini_mat==f_mat) = N1;
            b1 = sin(D1*pi*(k_ini_mat+f_mat))./sin(pi*(k_ini_mat+f_mat)/N);
            Re1 = (a1.*cos(phase1)+b1.*cos(phase1))*A;
            Im1 = (a1.*sin(phase1)-b1.*sin(phase1))*A;
            a2 = sin(D2*pi*(k_ini_mat-f_mat))./sin(pi*(k_ini_mat-f_mat)/N);
            a2(k_ini_mat==f_mat) = N2;
            b2 = sin(D2*pi*(k_ini_mat+f_mat))./sin(pi*(k_ini_mat+f_mat)/N);
            Re2 = (a2.*cos(phase2)+b2.*cos(phase2))*A;
            Im2 = (a2.*sin(phase2)-b2.*sin(phase2))*A;
            F = [Re1-2*real(X1);
                Im1-2*imag(X1);
                Re2-2*real(X2);
                Im2-2*imag(X2);];
    end

end