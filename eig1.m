function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% isMax = 1 ,取A的前c个最大特征值对应的特征向量
% isMax = 0 ，取A的前c个最小
if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
if issparse(A) && c<size(A,1)
    if isMax == 0
        [eigvec,eigval] = eigs(A,c,'sm');%升序 从小到大
    else
        [eigvec,eigval] = eigs(A,c,'lm');%降序 从大到小
    end;
    eigval_full=[];
else
    if c==size(A,1)
        A=full(A);
    end
    [v,d] = eig(A);  % eig() 求A的全部特征值组成的对角阵d，以及A的特征向量构成v的列向量
    d = diag(d);
    d = real(d);
    if isMax == 0
        [~, idx] = sort(d);
    else
        [~, idx] = sort(d,'descend');
    end;
    idx1 = idx(1:c);
    eigval = d(idx1);
    eigvec = real(v(:,idx1));
    
    eigval_full = d(idx);
end




