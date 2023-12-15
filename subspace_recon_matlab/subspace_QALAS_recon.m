%% Subspace QALAS Reconstruction

clear; close all; clc;

set(0,'DefaultFigureWindowStyle','docked')

b1_rec      = 1; % 1 (recon b1 data)
load_mask   = 0; % 1 (load pre-calculated undersampling masks)


%% set path

addpath utils

dicom_path  = '/path/to/dicom/';
dicom_name  = '/name/of/dicom';

data_path   = '/path/to/data/';
save_path   = '/path/to/save/';


%% load dataset

fprintf('loading kspace... ')
kspace  = readcfl(strcat(data_path,'kspace'));
coils   = readcfl(strcat(data_path,'coils'));

% normalize k-space to help with regularization value choice
kspace  = kspace / norm(kspace(:));
fprintf('done\n')

fprintf('loading dicom header... ');
di              = dicominfo(strcat(dicom_path,dicom_name));
turbo_factor    = di.EchoTrainLength;
esp             = di.RepetitionTime * 1e-3;
alpha_deg       = di.FlipAngle;
num_acqs        = 5;
fprintf('done\n')

if b1_rec == 1
    fprintf('loading b1 data... ');
    b1_map = readcfl(strcat(data_path,'b1'));
    fprintf('done\n')
end


%% retrospective undersampling

fprintf('retrospective undersampling... ');

[M,N,C,E] = size(kspace);

accelx  = 3;
accely  = 3;

if load_mask == 0
    mask    = zeros(size(kspace),'single');
    
    calib   = 10;
    ellipse = 1;
    pp      = 0;
    jitter  = 1e-2;
    
    for ee = 1:5
        mask_acq(:,:,ee) = sq(sum(kspace(:,:,1,(ee-1)*turbo_factor+1:ee*turbo_factor),4));
    end
    mask_acq = single(mask_acq~=0);
    
    for t = 1:5
        mask_p = vdPoisMex(M, N, 0.1*M, 0.1*N, ...
            accelx * (1+randn*jitter), accely * (1+randn*jitter), calib, ellipse, pp);
        
        mask(:,:,:,(t-1)*turbo_factor+1:t*turbo_factor) = repmat(mask_p,[1,1,C,turbo_factor]);
        fprintf('Actual accleration compared to sampled points: %.6f\n',sum(sum(mask_acq(:,:,t)))./sum(mask_p(:)));
    end
    writecfl(strcat(save_path,'mask','_R',num2str(accelx),'X',num2str(accely)),mask(:,:,1,:));
    
else
    mask = readcfl(strcat(data_path,'mask','_R',num2str(accelx),'X',num2str(accely)));
    mask = repmat(mask,[1,1,C,1]);
end
figure; imagesc(sq(sum(mask(:,:,1,:),4))); axis image; colormap jet;

kspace = kspace.*mask;
fprintf('done\n')


%% conventional reconstruction

[M,N,C,E]       = size(kspace);
lambdas_conv    = [1e-5];
types_conv      = {'wavelet'};
iters_conv      = [200];

parameters_conv = {};
ctr = 1;

parameters_conv{ctr}.type = 'cg';
ctr = ctr + 1;

for tt = 1:length(types_conv)
    for ll = 1:length(lambdas_conv)
        for ii = 1:length(iters_conv)
            parameters_conv{ctr}.type     = types_conv{tt};
            parameters_conv{ctr}.lambdas  = lambdas_conv(ll);
            parameters_conv{ctr}.iters    = iters_conv(ii);
            
            ctr = ctr + 1;
        end
    end
end

coils_bart  = reshape(coils,M,N,1,C);

recs_conv   = zeros(M,N,num_acqs,length(parameters_conv));

ctr = 1;

% CG reconstructions w/o reg
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
fprintf('conventional reconstruction, set: %d/%d\n',ctr,length(parameters_conv))
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')

for qq = 1:num_acqs
    ksp         = reshape(sum(kspace(:,:,:,(qq-1)*turbo_factor+1:qq*turbo_factor),4),M,N,1,C);
    coeff_old   = bart('pics -g -S -w 1 -i 20',reshape(ksp,M,N,1,C),coils_bart);
    recs_conv(:,:,qq,ctr) = coeff_old;
end
ctr = ctr+1;

% w/ reg
for tt = 1:length(types_conv)
    for ll = 1:length(lambdas_conv)
        for ii = 1:length(iters_conv)
            tic
            
            fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            fprintf('conventional reconstruction, set: %d/%d\n',ctr,length(parameters_conv))
            fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            
            if(strcmp(parameters_conv{ctr}.type, 'llr'))
                bartstr = sprintf('-R L:3:0:%f -i %d',lambdas_conv(ll),iters_conv(ii));
            else
                bartstr = sprintf('-l1 -r %f -i %d',lambdas_conv(ll),iters_conv(ii));
            end
            
            for qq = 1:num_acqs
                ksp = reshape(sum(kspace(:,:,:,(qq-1)*turbo_factor+1:qq*turbo_factor),4),M,N,1,C);
                coeff_old = bart(sprintf('pics -g -S -w 1 %s',bartstr), ksp,coils_bart);
                recs_conv(:,:,qq,ctr) = coeff_old;
            end
            
            ctr = ctr+1;
            toc
        end
    end
end
fprintf('done\n')


%% generating dictionaries for conventional and subspace mapping

TR                      = 4500e-3;
num_reps                = 5;
echo2use                = 1;
gap_between_readouts    = 900e-3;
time2relax_at_the_end   = 0;

t1_entries  = [300:5:3000,3100:100:5000];
t2_entries  = [10:1:100,102:2:200,204:10:400,420:20:500];

inv_eff     = 0.5:0.05:1.0;
inv_eff_sub = 0.5:0.05:1.0;
b1_val      = 0.65:0.05:1.35;
b1_val_sub  = 0.65:0.05:1.35;

T1_entries  = repmat(t1_entries.', [1,length(t2_entries)]).';
T1_entries  = T1_entries(:);

T2_entries  = repmat(t2_entries.', [1,length(t1_entries)]);
T2_entries  = T2_entries(:);

t1t2_lut    = cat(2, T1_entries, T2_entries);

% remove cases where T2>T1
idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) < t1t2_lut(t,2)
        idx = idx+1;
    end
end

t1t2_lut_prune = zeross([length(t1t2_lut) - idx, 2]);

idx = 0;
for t = 1:length(t1t2_lut)
    if t1t2_lut(t,1) >= t1t2_lut(t,2)
        idx = idx+1;
        t1t2_lut_prune(idx,:) = t1t2_lut(t,:);
    end 
end

fprintf('dictionary size: %d \n', length(t1t2_lut_prune)*length(b1_val)*length(inv_eff));

signal_conv_fit     = zeross([length(t1t2_lut_prune), num_acqs, length(b1_val), length(inv_eff)]);
signal_sub          = zeross([E, length(t1t2_lut_prune), length(b1_val_sub), length(inv_eff_sub)]);
signal_sub_fit      = zeross([E, length(t1t2_lut_prune), length(b1_val), length(inv_eff)]);

length_b1_val       = length(b1_val);
length_b1_val_sub   = length(b1_val_sub);
length_inv_eff      = length(inv_eff);
length_inv_eff_sub  = length(inv_eff_sub);

% parallel computing
delete(gcp('nocreate'))
c = parcluster('local');
total_cores = c.NumWorkers;
parpool(min(ceil(total_cores*.5), length(b1_val)))

% for subspace
cnt         = 0;
iniTime     = clock;
iniTime0    = clock;

parfor b1 = 1:length_b1_val_sub
    for ie = 1:length_inv_eff_sub
        cnt             = cnt + 1;
        [Mz, Mxy]       = sim_qalas_pd_b1_eff_T2_v0_YH(TR, alpha_deg, esp, ...
                            turbo_factor, t1t2_lut_prune(:,1)*1e-3, t1t2_lut_prune(:,2)*1e-3, ...
                            num_reps, echo2use, gap_between_readouts, ...
                            time2relax_at_the_end, b1_val_sub(b1), inv_eff_sub(ie));
        
        temp_sub        = Mxy(:,:,end);
        
        signal_sub(:,:,b1,ie)       = temp_sub;
        
        % activate when parfor is not used
%         if mod(cnt,floor(length_b1_val_sub*length_inv_eff/10)) < 1e-2
%             fprintf('generating dictionary: %.1f%%\n',cnt/(length_b1_val_sub*length_inv_eff)*100);
%             fprintf('elapsed time: %.1f sec\n\n',etime(clock,iniTime));
%             iniTime = clock;
%         end
    end
end
fprintf('total elapsed time: %.1f sec\n\n',etime(clock,iniTime0));

% for fitting
cnt         = 0;
iniTime1    = clock;
parfor b1 = 1:length_b1_val
    for ie = 1:length_inv_eff
        cnt             = cnt + 1;
        [Mz_,Mxy_]      = sim_qalas_pd_b1_eff_T2(TR, alpha_deg, esp, ...
                            turbo_factor, t1t2_lut_prune(:,1)*1e-3, t1t2_lut_prune(:,2)*1e-3, ...
                            num_reps, echo2use, gap_between_readouts, ...
                            time2relax_at_the_end, b1_val(b1), inv_eff(ie));
        [Mz, Mxy]       = sim_qalas_pd_b1_eff_T2_v0_YH(TR, alpha_deg, esp, ...
                            turbo_factor, t1t2_lut_prune(:,1)*1e-3, t1t2_lut_prune(:,2)*1e-3, ...
                            num_reps, echo2use, gap_between_readouts, ...
                            time2relax_at_the_end, b1_val(b1), inv_eff(ie));
        
        temp_conv       = abs(Mxy_(:,:,end).');
        temp_sub_fit    = abs(Mxy(:,:,end));
        
        for n = 1:size(temp_sub_fit,2)
            temp_conv(n,:)      = temp_conv(n,:) / sum(abs(temp_conv(n,:)).^2)^0.5;
            temp_sub_fit(:,n)   = temp_sub_fit(:,n) / sum(abs(temp_sub_fit(:,n)).^2)^0.5;
        end
        
        signal_conv_fit(:,:,b1,ie)  = temp_conv;
        signal_sub_fit(:,:,b1,ie)   = temp_sub_fit;
        
        % activate when parfor is not used
%         if mod(cnt,floor(length_b1_val*length_inv_eff/10)) < 1e-2
%             fprintf('generating dictionary: %.1f%%\n',cnt/(length_b1_val*length_inv_eff)*100);
%             fprintf('elapsed time: %.1f sec\n\n',etime(clock,iniTime));
%             iniTime = clock;
%         end
    end
end
delete(gcp('nocreate'))
fprintf('total elapsed time: %.1f sec\n\n',etime(clock,iniTime1));

% for subspace
signal_sub = reshape(signal_sub,[size(signal_sub,1), ...
                        size(signal_sub,2)*size(signal_sub,3)*size(signal_sub,4)])';
figure; plot(signal_sub(1:1e3:end,:)'); title('QALAS signals for subspace');

% for conventional fitting
length_dict = length(t1t2_lut_prune);
dict_cnv    = zeross([length_dict * length(inv_eff), num_acqs, length(b1_val)]);
for t = 1:length(inv_eff)
    dict_cnv(1 + (t-1)*length_dict : t*length_dict, :, :) = signal_conv_fit(:,:,:,t);
end
figure; plot(dict_cnv(1:1e3:end,:)'); title('QALAS dictionary for conventional fitting');

% for subspace fitting
length_dict     = length(t1t2_lut_prune);
dict_sub_fit    = zeross([length_dict * length(inv_eff), E, length(b1_val)]);
for t = 1:length(inv_eff)
    dict_sub_fit(1 + (t-1)*length_dict : t*length_dict, :, :) = permute(signal_sub_fit(:,:,:,t),[2,1,3]);
end
figure; plot(dict_sub_fit(1:1e3:end,:)'); title('QALAS dictionary for subspace fitting');
fprintf('done\n')


%% fitting for conventional reconstruction

fprintf('fitting for conventional reconstruction... ');
recs_conv_sel   = recs_conv;
estimate_pd_map = 1;
[T1_map_conv,T2_map_conv,PD_map_conv,IE_map_conv] = dict_fit_qalas_conv_recon(permute(recs_conv_sel,[1,2,4,3]), ...
                                                    dict_cnv, t1t2_lut_prune, estimate_pd_map, ...
                                                    b1_map, inv_eff, ...
                                                    TR, alpha_deg, esp, turbo_factor, ...
                                                    num_reps, echo2use, gap_between_readouts, ...
                                                    time2relax_at_the_end);
fprintf('done\n')


%% generating basis

K = 4;

fprintf('generating bases... ')
tic
[u,s,v] = svd(signal_sub','econ');
phi     = reshape(u(:,1:K),1,1,1,1,1,E,K);
writecfl('phi',phi)
toc
fprintf('done\n')


%% subspace reconstruction


lambdas_sub     = [2e-6]; % wavelet
types_sub       = {'wavelet'};
% lambdas_sub     = [1e-6]; % llr
% types_sub       = {'llr'};

iters_sub       = [2000];

parameters_sub  = {};
ctr = 1;

parameters_sub{ctr}.type = 'cg';
ctr = ctr + 1;

for tt = 1:length(types_sub)
    for ll = 1:length(lambdas_sub)
        for ii = 1:length(iters_sub)
            parameters_sub{ctr}.type     = types_sub{tt};
            parameters_sub{ctr}.lambdas  = lambdas_sub(ll);
            parameters_sub{ctr}.iters    = iters_sub(ii);

            ctr = ctr + 1;
        end
    end
end

recs_sub = zeros(M,N,E,length(parameters_sub));

ctr = 1;

% CG reconstructions w/o reg
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
fprintf('subspace reconstruction, set: %d/%d\n',ctr,length(parameters_sub))
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')

ksp = reshape(kspace,M,N,1,C,1,E);
coeff_sub = bart('pics -g -S -w 1 -B phi', ksp,coils_bart); % wavelet
% coeff_sub = bart('pics -d5 -S -w 1 -B phi', ksp,coils_bart); % LLR
recs_sub(:,:,:,ctr) = reshape((squeeze(phi)*reshape(squeeze(coeff_sub),M*N,K).').',M,N,E);

ctr = ctr+1;

% w/ reg
for tt = 1:length(types_sub)
    for ll = 1:length(lambdas_sub)
        for ii = 1:length(iters_sub)
            tic
            
            fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            fprintf('subspace reconstruction, set: %d/%d\n',ctr,length(parameters_sub))
            fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            
            if(strcmp(parameters_sub{ctr}.type, 'llr'))
                bartstr = sprintf('-R L:3:0:%f -i %d',lambdas_sub(ll),iters_sub(ii));
            else
                bartstr = sprintf('-l1 -r %f -i %d',lambdas_sub(ll),iters_sub(ii));
            end
            
            ksp = reshape(kspace,M,N,1,C,1,E);
            coeff_sub = bart(sprintf('pics -g -S -w 1 -B phi %s',bartstr), ... 
                ksp,coils_bart); % wavelet
            % coeff_sub = bart(sprintf('pics -d5 -S -w 1 -B phi %s',bartstr), ... 
            %     ksp,coils_bart); % LLR
            recs_sub(:,:,:,ctr) = reshape((squeeze(phi)*reshape(squeeze(coeff_sub),M*N,K).').',M,N,E);
            
            ctr = ctr+1;
            toc
        end
    end
end


%% fitting for subspace reconstruction

fprintf('fitting for subspace reconstruction... ');
recs_sub_sel    = sq(recs_sub);
estimate_pd_map = 1;
[T1_map_sub,T2_map_sub,PD_map_sub,IE_map_sub] = dict_fit_qalas_sub_recon(permute(recs_sub_sel,[1,2,4,3]), ...
                                                    dict_sub_fit, t1t2_lut_prune, estimate_pd_map, ...
                                                    b1_map, inv_eff, ...
                                                    TR, alpha_deg, esp, turbo_factor, ...
                                                    num_reps, echo2use, gap_between_readouts, ...
                                                    time2relax_at_the_end);
fprintf('done\n')

