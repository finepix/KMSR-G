%{ 
 input:
        X: original dataset with n*d 
        c_num: number of clusters 
        Y_standard: original Labels n*1
        NITR: max iteratios to run (default:20)

 output: Gsr4 : final indicator matrix, 
         result = [acc,nmi,purity]; 
%}

%% load data
clear;
data_name ='scale';
addpath('..\codes');
load(['datasets/', data_name, '.mat'])


NITR = 20;
c_num = length(unique(Y_standard));

%% input graph

A_tail = X*X'; 


%% KMSR model

% randomly Initialize F/Q
[n,d]=size(X);
Q = orth(rand(c_num,c_num));
F = orth(rand(n,c_num));

% Initialize Y
% Initialize Y according to [(F0*F0')^(-1/2)]*F0,the index of maximum in F0'rows is set to 1£¬
% where F0 is the top smallest c_num eigenvectors of laplas.
options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'HeatKernel';
options.k = 5;  
W = constructW(X,options);
A = full(W);
D = diag(sum(A,2));
laplas = full(D-A);
[F0,~,~] = eig1(laplas,c_num,0,1); 
Y = diag(diag(F0*F0'))^(-0.5)*F0; 
[~, g] = max(Y,[],2);
Y = TransformL(g, c_num);

%% parameter tuning
tt = -3:0.5:3;
lambda = power(10,tt);
% lamd = 0.1:0.04:1;
% lamd = 0.51:0.01:0.81;
len_t=length(lambda);
obj_kmsr_func=zeros(NITR,len_t);  %record obj_kmsr under different value of lambda.
changed = zeros(len_t,NITR); %record iteration number for updating Y.
result = zeros(len_t,3);
record_kmv = zeros(len_t,1);

for r = 1:len_t
    Gsr4 = Y;
    Q_0 = Q;
    F_0 = F;

    %% Optimization 
    for iter = 1:NITR
       %% Update F 
        M_gpi = Gsr4*(Gsr4'*Gsr4+eps*eye(c_num))^-0.5;  
        B = M_gpi*Q_0'; 
        err_mean=1;
        t_gpi=1;
        while err_mean>1e-5
            M_F = 2*A_tail*F_0+2*lambda(r)*B; % GPI-based  method
            [U_F,~,V_F] = svd(M_F,'econ');
            F_0 = U_F*V_F'; 
            clear U_F V_F;
            obj_gpi(t_gpi) = trace(F_0'*A_tail*F_0)+2*lambda(r)*trace(F_0'*B);   
            if t_gpi>=3
                mean1 = mean(obj_gpi(t_gpi-2)+obj_gpi(t_gpi-1));
                mean2 = mean(obj_gpi(t_gpi-1)+obj_gpi(t_gpi));
                err_mean = abs(mean2 - mean1);
            end
            t_gpi = t_gpi+1;
            if t_gpi > 100
                break;
            end
        end
        
       %% Update Q 
        M_q = (Gsr4'*Gsr4+eps*eye(c_num))^-0.5;  
        M_Q = M_q * Gsr4';
        [U_Q, ~, V_Q] = svd(M_Q*F_0); 
        Q_0 = V_Q*U_Q';
        clear V_Q U_Q; 
        
       %% Update Y 
        G = F_0*Q_0;
        yg = diag(Gsr4'*G)';
        yy = diag(Gsr4'*Gsr4+ eps*eye(c_num))';
        for it = 1:10
            converged=true;
            for i = 1:n    % solve Y row by row
                gi = G(i,:);
                yi = Gsr4(i,:);
                [~,id0] = max(yi);
                ss = (yg+gi.*(1-yi))./sqrt(yy+1-yi) - (yg-gi.*yi)./sqrt(yy-yi);
                [~,id] = max(ss(:));
                if id~=id0
                    converged=false;
                    changed(r,iter)=changed(r,iter)+1;
                    %update Gsr4
                    yi = zeros(1,c_num);
                    yi(id) = 1;
                    Gsr4(i,:) = yi;
                    
                    %update yy
                    yy(id0) = yy(id0) - 1;  
                    yy(id) = yy(id) + 1;
                    
                    %update yg
                    yg(id0) = yg(id0) - gi(id0); 
                    yg(id) = yg(id) + gi(id);
                end;
            end;
            if converged
                clear s nn;
                break;
            end
            
        end;
        
       %% calculate obj_KMSR
        temp1 = (-1)*trace(F_0'*(X*X')*F_0);
        r1 = Gsr4*(Gsr4'*Gsr4+eps*eye(c_num))^-0.5;  
        G = F_0*Q_0;
        temp2 = trace((G-r1)'*(G-r1)); 
        obj_kmsr_func(iter,r) = temp1+lambda(r)*temp2;  
        
       %% convergence judgement
        if iter>2 && abs(obj_kmsr_func(iter,r)-obj_kmsr_func(iter-1,r))<1e-6
            break; 
        end
    end
    record_kmv(r)=obj_kmsr_func(iter,r);
    
    %% clustering result
    Gsr4_col = nc2n(Gsr4);
    result(r,:)= ClusteringMeasure(Y_standard,Gsr4_col);
    
    %% Save
%     intergrate=floor(lamd(r));
%     if lamd(r) == intergrate
%         filename = ['D:/Clustering/kmsr/result-save-X-X/ecoli/', data_name, '_' num2str(lamd(r))];
%     else
%         code=num2str(lamd(r)-intergrate);
%         code_length=length(code);
%         subcode=code(3:code_length);
%         filename = ['D:/Clustering/kmsr/result-save-X-X/ecoli/', data_name, '_' num2str(intergrate)  '_' subcode];
%     end
%     save(filename,'F','Y','F_0','Q_0','Gsr4','record_kmv','totalD','result');
    
end

result