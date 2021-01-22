function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% isMax = 1 ,ȡA��ǰc���������ֵ��Ӧ����������
% isMax = 0 ��ȡA��ǰc����С
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
        [eigvec,eigval] = eigs(A,c,'sm');%���� ��С����
    else
        [eigvec,eigval] = eigs(A,c,'lm');%���� �Ӵ�С
    end;
    eigval_full=[];
else
    if c==size(A,1)
        A=full(A);
    end
    [v,d] = eig(A);  % eig() ��A��ȫ������ֵ��ɵĶԽ���d���Լ�A��������������v��������
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




