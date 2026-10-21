%% ===============================================================
%     ⚡ UNIVERSAL DIGITAL MODULATION OVER AWGN & RAYLEIGH CHANNELS
%     - Supports: BPSK | QPSK | 8PSK | 16QAM | 32QAM | 64QAM
%     - Gray-coded mapping with ML detection
%     - Common Eb/N0 range for fair comparison
%     - Transmitted vs received constellation comparison
% ===============================================================
clc; clear; close all;

%% ===============================================================
%   🚀 USER CONFIGURATION
% ===============================================================
mod_type = upper(input('Enter modulation type (BPSK/QPSK/8PSK/16QAM/32QAM/64QAM): ','s'));
valid_mods = {'BPSK','QPSK','8PSK','16QAM','32QAM','64QAM'};
if ~ismember(mod_type, valid_mods)
    warning('Invalid modulation! Defaulting to 16QAM.');
    mod_type = '16QAM';
end

Nsym = 3e5;
EbN0_dB = 1:1:30;
numRuns = length(EbN0_dB);
batchSize = 1e5;

switch mod_type
    case 'BPSK',   M = 2;
    case 'QPSK',   M = 4;
    case '8PSK',   M = 8;
    case '16QAM',  M = 16;
    case '32QAM',  M = 32;
    case '64QAM',  M = 64;
end
k = log2(M);
Nbits = Nsym * k;

fprintf('\n================= SIMULATION CONFIGURATION =================\n');
fprintf('Selected Modulation : %s\n', mod_type);
fprintf('Modulation Order    : M = %d (%.0f bits/symbol)\n', M, k);
fprintf('Total Symbols       : %.0f (%.0f bits)\n', Nsym, Nbits);
fprintf('Eb/N0 Range (dB)    : %.1f → %.1f (%d points)\n', EbN0_dB(1), EbN0_dB(end), numRuns);
fprintf('Processing Batch    : %.0f symbols/iteration\n', batchSize);
fprintf('============================================================\n\n');

%% ===============================================================
%   🧭 CONSTELLATION & GRAY MAPPING GENERATION
% ===============================================================
symList = (0:M-1)';
if contains(mod_type, 'PSK')
    if M == 2
        constellation = pskmod(symList, M, 0, 'gray');
    else
        constellation = pskmod(symList, M, pi/M, 'gray');
    end
else
    % Handles rectangular (16QAM, 64QAM) and cross (32QAM) constellations
    constellation = qammod(symList, M, 'gray', 'UnitAveragePower', true);
end

% Normalize average symbol energy to Es = 1
constellation = constellation / sqrt(mean(abs(constellation).^2));
Es = mean(abs(constellation).^2);
fprintf('Constellation generated and normalized (Es = %.4f)\n', Es);

% Precompute Gray bit table for bit error tracking
bitTable = de2bi(symList, k, 'left-msb'); 

%% ===============================================================
%   🎯 MONTE CARLO SIMULATION
% ===============================================================
BER_AWGN     = zeros(1, numRuns);
BER_Rayleigh = zeros(1, numRuns);
SER_AWGN     = zeros(1, numRuns);
SER_Rayleigh = zeros(1, numRuns);

snapshot_EbN0 = [15, 25];
snapshot_data = repmat(struct('tx',[],'rx_awgn',[],'rx_ray',[]), numRuns, 1);

fprintf('--- Starting Monte Carlo Simulation ---\n');

for idxE = 1:numRuns
    EbN0_lin = 10^(EbN0_dB(idxE)/10);
    EsN0 = EbN0_lin * k;
    noiseVar = 1 / (2 * EsN0);
    
    bitErr_awgn = 0; bitErr_ray = 0;
    symErr_awgn = 0; symErr_ray = 0;
    processed = 0;
    
    while processed < Nsym
        idxEnd = min(processed + batchSize, Nsym);
        B = idxEnd - processed;
        processed = idxEnd;
        
        % Transmitter
        txSymIdx = randi([0, M-1], B, 1);
        txSym = constellation(txSymIdx + 1);
        txBits = bitTable(txSymIdx + 1, :);
        
        % Channel
        noise = sqrt(noiseVar) * (randn(B,1) + 1i*randn(B,1));
        rx_awgn = txSym + noise;
        
        h = (randn(B,1) + 1i*randn(B,1)) / sqrt(2);
        rx_ray = h .* txSym + noise;
        rx_ray_eq = rx_ray ./ h;   % Zero-Forcing equalization
        
        % Snapshot collection
        if ismember(EbN0_dB(idxE), snapshot_EbN0)
            nSnap = min(4000, B);
            snapshot_data(idxE).tx      = [snapshot_data(idxE).tx; txSym(1:nSnap)];
            snapshot_data(idxE).rx_awgn = [snapshot_data(idxE).rx_awgn; rx_awgn(1:nSnap)];
            snapshot_data(idxE).rx_ray  = [snapshot_data(idxE).rx_ray; rx_ray_eq(1:nSnap)];
        end
        
        % Minimum Euclidean Distance Detection (ML)
        % Using pdist2-equivalent vector distance for memory efficiency
        dist_awgn = abs(rx_awgn - constellation.');
        [~, minIdx_awgn] = min(dist_awgn, [], 2);
        decSym_awgn = minIdx_awgn - 1;
        
        dist_ray = abs(rx_ray_eq - constellation.');
        [~, minIdx_ray] = min(dist_ray, [], 2);
        decSym_ray = minIdx_ray - 1;
        
        % Symbol Errors
        symErr_awgn = symErr_awgn + sum(decSym_awgn ~= txSymIdx);
        symErr_ray  = symErr_ray  + sum(decSym_ray  ~= txSymIdx);
        
        % Bit Errors
        rxBits_awgn = bitTable(decSym_awgn + 1, :);
        rxBits_ray  = bitTable(decSym_ray + 1, :);
        bitErr_awgn = bitErr_awgn + sum(sum(rxBits_awgn ~= txBits));
        bitErr_ray  = bitErr_ray  + sum(sum(rxBits_ray  ~= txBits));
    end
    
    BER_AWGN(idxE)     = bitErr_awgn / (Nsym * k);
    BER_Rayleigh(idxE) = bitErr_ray / (Nsym * k);
    SER_AWGN(idxE)     = symErr_awgn / Nsym;
    SER_Rayleigh(idxE) = symErr_ray / Nsym;
    
    fprintf('Eb/N0 = %2.0f dB → BER(AWGN)=%.3e, BER(Rayleigh)=%.3e\n', ...
        EbN0_dB(idxE), BER_AWGN(idxE), BER_Rayleigh(idxE));
end

%% ===============================================================
%   📊 PLOTS
% ===============================================================
% BER Curves
figure('Name', sprintf('BER Performance of %s', mod_type));
semilogy(EbN0_dB, BER_AWGN, 'b-o', 'LineWidth', 1.5); hold on;
semilogy(EbN0_dB, BER_Rayleigh, 'r-s', 'LineWidth', 1.5);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('Bit Error Rate (BER)');
title(sprintf('BER Curve: %s under AWGN & Rayleigh', mod_type));
legend('AWGN', 'Rayleigh', 'Location', 'southwest');

% SER Curves
figure('Name', sprintf('SER Performance of %s', mod_type));
semilogy(EbN0_dB, SER_AWGN, 'b-o', 'LineWidth', 1.5); hold on;
semilogy(EbN0_dB, SER_Rayleigh, 'r-s', 'LineWidth', 1.5);
grid on; xlabel('E_b/N_0 (dB)'); ylabel('Symbol Error Rate (SER)');
title(sprintf('SER Curve: %s under AWGN & Rayleigh', mod_type));
legend('AWGN', 'Rayleigh', 'Location', 'southwest');

% Ideal Constellation Plot
figure('Name', 'Ideal Constellation');
plot(real(constellation), imag(constellation), 'bo', 'MarkerFaceColor', 'b');
title(sprintf('Ideal %s Constellation Diagram', mod_type));
xlabel('In-phase (I)'); ylabel('Quadrature (Q)');
grid on; axis equal;

%% ===============================================================
%   🌌 TRANSMITTED VS RECEIVED CONSTELLATION COMPARISON
% ===============================================================
for s = 1:length(snapshot_EbN0)
    targetSNR = snapshot_EbN0(s);
    idx = find(EbN0_dB == targetSNR, 1);
    if isempty(idx) || isempty(snapshot_data(idx).tx), continue; end
    
    tx_snap      = snapshot_data(idx).tx;
    rx_awgn_snap = snapshot_data(idx).rx_awgn;
    rx_ray_snap  = snapshot_data(idx).rx_ray;
    
    figure('Name', sprintf('Constellation Comparison @ %.0f dB', targetSNR), 'NumberTitle', 'off');
    subplot(1,3,1);
    plot(real(tx_snap), imag(tx_snap), 'bo', 'MarkerSize', 4); grid on; axis equal;
    title(sprintf('Transmitted (%s)', mod_type));
    xlabel('In-phase (I)'); ylabel('Quadrature (Q)');
    
    subplot(1,3,2);
    plot(real(rx_awgn_snap), imag(rx_awgn_snap), 'r.', 'MarkerSize', 5); grid on; axis equal;
    title(sprintf('Received AWGN @ %.0f dB', targetSNR));
    xlabel('In-phase (I)'); ylabel('Quadrature (Q)');
    
    subplot(1,3,3);
    plot(real(rx_ray_snap), imag(rx_ray_snap), 'g.', 'MarkerSize', 5); grid on; axis equal;
    title(sprintf('Received Rayleigh @ %.0f dB', targetSNR));
    xlabel('In-phase (I)'); ylabel('Quadrature (Q)');
end

%% ===============================================================
%   🧾 LAB REPORT SUMMARY
% ===============================================================
fprintf('\n================== LAB REPORT SUMMARY ==================\n');
fprintf('Modulation Scheme  : %s\n', mod_type);
fprintf('Constellation Size : %d\n', M);
fprintf('Bit/Symbol Ratio   : %.0f bits/symbol\n', k);
[minBERawgn, ia] = min(BER_AWGN);
[minBERray, ib]  = min(BER_Rayleigh);
fprintf('Best BER (AWGN)    : %.3e @ %.1f dB\n', minBERawgn, EbN0_dB(ia));
fprintf('Best BER (Rayleigh): %.3e @ %.1f dB\n', minBERray, EbN0_dB(ib));
fprintf('Average SER (AWGN) : %.3e\n', mean(SER_AWGN));
fprintf('Average SER (Ray.) : %.3e\n', mean(SER_Rayleigh));
fprintf('=========================================================\n\n');