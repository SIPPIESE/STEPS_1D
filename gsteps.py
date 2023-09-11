import numpy as np
import subprocess
import glob
import time
import pandas

class stepsPeak:
    def __init__ (self):
        pass
        

    def BIGPEAK(self, absfftshiftsignal, num_peaks):
        # Extracts certain number (num_peaks) of macro-scale peaks in absfftshiftsignal. Merge clustering peaks. 
        # Returns indicies of the local-maximum peaks, one per cluster. Higher peaks are prioritized.
        tolmultiplier = 2
        fulllen = np.size(absfftshiftsignal, 0)
        tolrange = round(fulllen/2560)
        if (tolrange < 3):
            tolrange = 3
        midpoint = round(fulllen / 2 + 0.5)
        halfsignal = absfftshiftsignal[midpoint + 1:]
        std_baseline = tolmultiplier * np.std(halfsignal)

        avg = np.mean(halfsignal)
        indicies = []
        for i in np.arange(num_peaks):
            curr_index = np.argmax(halfsignal)
            indicies.append(curr_index)

            j = 0
            while (curr_index - j - tolrange) > 0 and (curr_index + j + tolrange) - np.size(halfsignal, 0) < -1:  # decide peak leakage range
                if abs(np.mean(halfsignal[curr_index + j: curr_index + j + tolrange]) - avg) < std_baseline:
                    break
                j = j + 1
            indend = curr_index + j + tolrange
            j = 0
            while (curr_index - j - tolrange) > 0 and (curr_index + j + tolrange) - np.size(halfsignal, 0) < -1:
                if abs(np.mean(halfsignal[curr_index - j - tolrange: curr_index - j]) - avg) < std_baseline:
                    break
                j = j + 1
            indbeg = curr_index - j - tolrange

            halfsize = np.size(halfsignal, 0)
            if indbeg < 0:
                indbeg = 1
            if indend > halfsize:
                indend = halfsize

            halfsignal[indbeg: indend] = avg - std_baseline       
            avg = np.mean(halfsignal)

        indicies = np.sort(indicies, axis=0)
        return indicies


    def SMALLPEAK(self, absfftshiftsignal, peakindice, BLMP = 2):
        # Inspects the peak index given by BIGPEAK() method close-in. 
        # Will search for all small peaks in the same cluster as in BIGPEAK(). Decrease "BLMP" to allow more extraction. 
        tolmultiplier = 0.5
        fulllen = np.size(absfftshiftsignal, 0)
        tolrange = round(fulllen/2560)
        if (tolrange < 3):
            tolrange = 3
        midpoint = round(fulllen / 2 + 0.5)
        halfsignal = absfftshiftsignal[midpoint: fulllen]
        std_baseline = tolmultiplier * np.std(halfsignal[1:])
        avg = np.mean(halfsignal)
        indicies = []
        curr_index = peakindice

        j = 0
        while (curr_index - j - tolrange) > 0 and (curr_index + j + tolrange) - np.size(halfsignal, 0) < -1: 
                if abs(np.mean(halfsignal[curr_index + j: curr_index + j + tolrange]) - avg) < std_baseline: 
                    break
                j = j + 1
        indend = curr_index + j + tolrange
        j = 0
        while (curr_index - j - tolrange) > 0 and (curr_index + j + tolrange) - np.size(halfsignal, 0) < -1: 
            if abs(np.mean(halfsignal[curr_index - j - tolrange: curr_index - j]) - avg) < std_baseline: 
                break
            j = j + 1
        indbeg = curr_index - j - tolrange

        if indbeg < 1:
            indbeg = 1
        indicies = self.stdBaseline(halfsignal[indbeg: indend], BLMP)
        indicies = indicies + indbeg - 1
        return np.asarray(indicies)


    def stdBaseline(self, signal, baseline_multiplier):
        # Finds peaks that exceeds a baseline in all entries of signal.
        # The baseline is proportional to standard deviation by coefficient baseline_multiplier.
        indicies = []
        length = np.size(signal, 0)
        avg_signal = np.mean(signal)
        std_signal = np.std(signal)
        # print("std = ", std_signal)
        baseline = baseline_multiplier * std_signal
        # print("baseline = ", baseline)
        size_indicies = 0   
        curr_signal = signal[0]
        if curr_signal > signal[1] and curr_signal - avg_signal > baseline:
            indicies.append(0)
        for i in np.arange(1, length - 1): 
            curr_signal = signal[i]
            if curr_signal - avg_signal > baseline and curr_signal > signal[i - 1] and curr_signal > signal[i + 1]:
                # print("curr = ", curr_signal)
                indicies.append(i)
                size_indicies = size_indicies + 1
        curr_signal = signal[length - 1]
        if curr_signal > signal[length - 2] and curr_signal - avg_signal > baseline: 
            indicies.append(length - 1)
        # print("indicies = ", indicies)
        return indicies
    
    def gsteps(self, x0, k0):
        N = np.size(x0)

        # get initial values by ipDFT
        n = np.arange(0, N)
        hannwindow = np.hanning(N + 1)[0: -1]  # matlab periodic hann
        x = np.multiply(x0, hannwindow)
        M = np.size(k0, 0)
        X1 = np.zeros(1, M)
        coeff_common_1 = -1j * 2 * np.pi * n
        X_est_0 = np.matmul(x, np.exp(np.matmul(coeff_common_1, k0) / N))
        X1 = X_est_0
        X_est_l = np.matmul(x, np.exp(np.matmul(coeff_common_1, (k0 - 1)) / N))
        X_est_u = np.matmul(x, np.exp(np.matmul(coeff_common_1, (k0 + 1)) / N))
        Mag0 = np.abs(X_est_0)
        Magl = np.abs(X_est_l)
        Magu = np.abs(X_est_u)
        delta = np.divide(2 * (Magu - Magl), (Magu + 2 * Mag0 + Magl))
        k_ini = k0 + delta
        W_d_en1 = np.multiply(np.divide(2*np.sin(np.pi*delta), np.sin(np.pi*delta/N)), np.exp(1j*np.pi*(N-1)/N*delta))
        W_d_en2 = np.multiply(np.divide(np.sin(np.pi*(1-delta)), np.sin(np.pi*(1-delta)/N)), np.exp(-1j*np.pi*(N-1)/N*(1-delta)))
        W_d_en3 = np.multiply(np.divide(np.sin(np.pi*(1+delta)), np.sin(np.pi*(1+delta)/N)), np.exp(1j*np.pi*(N-1)/N*(1+delta)))
        W_d = np.abs(W_d_en1 - W_d_en2 - W_d_en3)

        for index in np.arange(np.size(W_d)):  # obliterate all 0
            if W_d[index] == 0:
                W_d[index] = 2 * N
    
        A_ini = np.divide(8*abs(X_est_0), (W_d))
        row_ones = np.ones(M,1)
        col_ones = np.ones(1,M)
        k_ini_mat = np.matmul(np.transpose(k_ini), row_ones)

        # initial frequency
        k_ini1 = k_ini[0]
        if np.mod(k_ini1, 2) != 0:
            coef2 = ((np.floor(k_ini1)+1)/2)/(k_ini1)
        else:
            coef2 = (np.floor(k_ini1)/2)/(k_ini1)
        # % coef2 = floor(k_ini1)/k_ini1;
        D1 = 1
        D2 = np.round(N*coef2)/N
        N1 = np.round(D1*N)
        N2 = np.round(D2*N)
        if np.mod(N2,2) != 0:
            N2 = N2 - 1
            D2 = N2 / N
        ns1 = np.arange(-(N1-1)/2, (N1-1)/2 + 1)
        x1 = x0[:N1]
        X1 = np.matmul(x1, np.exp(-1j*2*np.pi/N1*np.matrix.H(ns1)*D1*k_ini))
        phi_ini = np.angle(X1)
        X1 = np.transpose(X1)
        ns2 = np.arange(-(N2-1)/2, (N2-1)/2 + 1)
        x2 = x0[np.round(N*(1-D2)) :]
        X2 = np.matmul(x2, np.exp(-1j * 2 * np.pi / N2 * np.matrix.H(ns2) * D2 * k_ini))
        X2 = np.transpose(X2)
        t0 = np.transpose(np.array([k_ini, A_ini, phi_ini]))

        # solve nonlinear equations
        fun = @(t)nlEqu(t);
        [t_hat, ~] = fsolve(fun, t0, options);
        # % f_est = t_hat(1, 1:9);
        # % A_est = t_hat(2, 1:9);
        # % phi_est = t_hat(3, 1:9);
        f_est = t_hat(1);
        A_est = t_hat(2);
        phi_est = t_hat(3);

            # objective function
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
        return f_est, A_est, phi_est
    
    
if __name__ == '__main__':
    # headerlines = 5
    # num_peaks = 20
    # str_headerlines = str(headerlines)
    # filelist = glob.glob('input/*.xlsx')
    # for filename in filelist:
    #     exact_name = filename.split('\\')[1]
    #     save_filename = 'output/STEPS_estimated_' + exact_name.removesuffix('.xlsx') + '.txt'
    #     fcreate = open(save_filename, 'w', encoding='gbk')
    #     fcreate.close()

    #     demo_signal = np.array(pandas.read_excel(filename, header=headerlines))
    #     demo = stepsPeak()
    #     fftsignal = np.abs(np.fft.fftshift(np.fft.fft(demo_signal[:, 1])))
    #     big_peak_int = demo.BIGPEAK(fftsignal, num_peaks)