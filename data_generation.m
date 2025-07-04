% Data Generation Script for GPT-based ML
% Clear workspace and set up path
clear; close all; clc;
addpath('../../');

%% Parameters (EDIT THESE)
noiseLevel   = 0.1;                 % noise standard deviation
dictLength   = 6;                   % number of dictionary shapes
nSamples     = 2;                  % number of random samples per shape
ord          = 1;                   % GPT order (1,2,3,...)
nbPoints     = 2^10;                % boundary discretization points
freqlist     = linspace(100,200,3); % frequencies to evaluate CGPTs
delta        = 1;                   % reference diameter for shapes

%% Build dictionary of reference shapes and names
D = cell(1, dictLength);
%shapeNames = {'flower5','ellipse', 'rectangle'};
shapeNames = {'flower5', 'flower6', 'ellipse', 'circle', 'rectangle', 'triangle'};
D{1} = shape.Flower(delta/2, delta/2, nbPoints, 5, 0.4, 0);
D{2} = shape.Flower(delta/2, delta/2, nbPoints, 6, 0.4, 0);
D{3} = shape.Ellipse(delta, delta/2, nbPoints);
D{4} = shape.Ellipse(delta, delta,   nbPoints);
D{5} = shape.Rectangle(delta, delta/2, nbPoints);
D{6} = shape.Triangle(delta, delta/2, nbPoints);
%D{7} = shape.Triangle(delta, delta, nbPoints);
%D{8} = shape.Flower(delta, delta/2, nbPoints, 4, 0.4, 0);
%D{9} = shape.Rectangle(delta, delta, nbPoints);
%D{10} = shape.Triangle(4*delta, delta/2, nbPoints);
%D{11} = shape.Triangle(4*delta, delta/2, nbPoints);
%D{12} = shape.Triangle(4*delta, delta/2, nbPoints);
%D{13} = shape.Triangle(4*delta, delta/2, nbPoints);
%D{14} = shape.Triangle(4*delta, delta/2, nbPoints);








%% Set up the fish acquisition environment (once)
mcenter = [0;0];
mradius = 5*delta;  % measurement circle radius
Omega   = shape.Banana(mradius*delta*2.5, mradius*delta/10, [mradius;0], 0, 0, nbPoints/2);
idxRcv  = 1:2:Omega.nbPoints;        % receptor indices on skin
impd    = 0.001;                     % skin impedance
cfg     = acq.Fish_circle(Omega, idxRcv, mcenter, mradius, 10, 2*pi, [], [], 0.5, impd);
stepBEM = 4;                         % downsampling factor for P1-BEM basis


% Open CSV file for writing
dataFile = 'data-generation.csv';
fid = fopen(dataFile,'w');
% Write CSV header
fprintf(fid,'shape,frequency,order,type,instance,CGPT_entries\n');

%% Loop through dictionary shapes and samples
for rep = 1:1
    for k = 1:dictLength
        for s = 1:nSamples
            fprintf('\n=== %s, Sample #%d ===\n', shapeNames{k}, s);
            base = D{k};
            % random baseline transform on shape: rotation, uniform scale [0.5,1.5], translation [-delta/3,delta/3]
            theta = 2*pi*rand();
            theta2 = 2*pi*rand();
            
            scale = 0.5 + rand()*1.0;  % can shrink or enlarge
            scale2 = 0.5 + rand()*1.0;  % can shrink or enlarge
            trans = delta/3*(rand(2,1)*2 - 1);
            trans2 = delta/3*(rand(2,1)*2 - 1);
            S_base = (base < theta)*scale + trans;
            S_base2 = (base < theta2)*scale2 + trans2;
            instances = { S_base, S_base2 };
            instNames = {'rand1','rand2'};
            
            % use same params for all
            freqs = freqlist;
            cnd   = [10];  pmtt = [0.1];
            
            % Loop over instances (here only baseline)
            for j = 1:numel(instances)
                fprintf('\n-- %s Instance --\n', instNames{j});
                curShape = instances{j};
    
                % 1) Print theoretical CGPTs
                for f = 1:numel(freqs)
                    omega  = freqs(f);
                    lambda = asymp.CGPT.lambda(cnd, pmtt, omega);
                    Mth    = asymp.CGPT.theoretical_CGPT({curShape}, lambda, ord);
                    entries = reshape(Mth(1:2*ord,1:2*ord).',1,[]);
                    fprintf(fid, '%s,%.2f,%d,theoretical,%s,"%s"\n', ...
                    shapeNames{k}, omega, ord, instNames{j}, num2str(entries,' %g'));
    
                    fprintf('Theoretical CGPT (order=%d) at ω=%.1f:\n', ord, omega);
                    disp(Mth(1:2*ord,1:2*ord));
                end
    
                % 2) Build solver for this single inclusion
                P = PDE.ElectricFish({curShape}, cnd, pmtt, cfg, stepBEM);
    
                % 3) Simulate & add noise
                data = P.data_simulation(freqs);
                data = P.add_white_noise(data, noiseLevel);
    
                % 4) Reconstruct and PRINT CGPTs
                fprintf('\n-- Reconstruction --\n');
                for f = 1:numel(freqs)
                    omega = freqs(f);
                    MSR   = data.MSR_noisy{f}; Cur = data.Current_noisy{f};
                    out   = P.reconstruct_CGPT(MSR, Cur, ord, 10000, 1e-10, 0);
    
                    Mrec  = out.CGPT;
                    entries = reshape(Mrec(1:2*ord,1:2*ord).',1,[]);
                    fprintf(fid, '%s,%.2f,%d,reconstructed,%s,"%s"\n', ...
                         shapeNames{k}, omega, ord, instNames{j}, num2str(entries,' %g'));
                    fprintf('Reconstructed CGPT (order=%d) at ω=%.1f:\n', ord, omega);
                    disp(out.CGPT);
                end
            end
        end
    end
end