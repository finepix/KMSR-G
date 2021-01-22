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

%% construct (normalized) affinity matrix
% Heatkernel method
options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'HeatKernel';
options.k = 5;   
W = constructW(X,options);
A = full(W);

% constuct affinity matix by CAN.
% ck = 5;
% [~, A, ~] = CAN(X', c_num,ck);

% constuct affinity matix by PCAN.
% pd = c_num-1; 
% pk = 5;
% [~, ~, A, ~] = PCAN(X', c_num,pd,pk);

%% compute A_tail.
D = diag(sum(A,2));
laplas = full(D-A);
D2 = diag(diag(D).^(-0.5));
A_tail = D2*A*D2;

%% KMSR-G model
% Initialize Y according to [(F0*F0')^(-1/2)]*F0,the index of maximum in F0'rows is set to 1£¬
% where F0 is the top smallest c_num eigenvectors of laplas.
[n,d] = size(X);
[F0,~,~] = eig1(laplas,c_num,0,1); 
Y = diag(diag(F0*F0'))^(-0.5)*F0; 
[~, g] = max(Y,[],2);
Y = TransformL(g, c_num);

% Initialize Q randomly.
% rand('twister',5489);
Q = orth(rand(c_num,c_num));

%% tuning parameter
tt = -3:0.5:3;
lamd = power(10,tt);
% lamd = 0.1
% lamd = 0.0001:0.0001:0.003;
% lamd = 0.003:0.001:0.031;
% lamd = 0.1:0.01:0.31;
% lamd = 0.3:0.03:1;
% lamd = 1.1:0.5:10.1;

len_t = length(lamd);
obj_func_rcisr = zeros(NITR,len_t);  
record_kmv = zeros(len_t,1);
result = zeros(len_t,3);
changed = zeros(len_t,1);
itt = zeros(len_t,1); %count the number iteration of solving Y under certain lambda.
tic; 
for r = 1:len_t
    Gsr4 = Y;
    Q_0 = Q;
    F_0 = F0;
   
    for iter = 1:NITR
       %% update F 
        [F_0,runtimes,obj_gpi] = main2_updateF(A_tail,Gsr4,c_num,F_0,Q_0,lamd(r));
        
        
       %% update Q 
        M_q = (Gsr4'*Gsr4+eps*eye(c_num))^(-0.5);  
        M_Q = M_q * Gsr4';
        [U_Q, ~, V_Q] = svd(M_Q*F_0);  
        Q_0 = V_Q*U_Q';
        clear V_Q U_Q;
        
        
       %% update Y 
        [Gsr4,changed(iter),itt(iter)] = main2_updateY(F_0,Q_0,Gsr4);
        
       %% calculate obj_KMSR-G
        temp1 = (-1)*trace(F_0'*A_tail*F_0);
        r1_temp = (Gsr4'*Gsr4+eps*eye(c_num))^(-0.5);  
        r1 = Gsr4 * r1_temp;
        temp2 = trace((F_0*Q_0-r1)'*(F_0*Q_0-r1)); 
        obj_func_rcisr(iter,r) = temp1+lamd(r)*temp2;  
        
       %% convergence judgement
        if iter>=2 && abs(obj_func_rcisr(iter,r)-obj_func_rcisr(iter-1,r))<1e-8
            break; 
        end
    end
    record_kmv(r)=obj_func_rcisr(iter,r);
    
    %% clustering performance: acc/nmi/purity
     Gsr4_col = nc2n(Gsr4);
     result(r,:)= ClusteringMeasure(Y_standard,Gsr4_col);
    
end
%plot(totalD);
ttt=toc;
disp(['iteration run for ' num2str(ttt) ' second']);
result
ans = lamd';