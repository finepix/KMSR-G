function Y = TransformL(y, nclass, type)
% [~, g] = max(F,[],2); % g是n*1矩阵，每个值为F每行最大值的下标位置
% Gsr4 = TransformL(g, class_num);
% TransformL()将原矩阵F中每行最大值的位置取1，其余取0，构成Y
% 将连续的标识矩阵F 一行中最大值的位置 转化为 0-1 标识矩阵Y
% y是一个列向量，yi是原矩阵每行最大值的列下标

% 等价于：将矩阵Y中每行最大值的位置取1，其余取0
% for i = 1:n
%     [~,mix] = max(Y(i,:));
%     Y(i,:) = 0;
%     Y(i,mix) = 1;
% end



n =length(y);  %原矩阵的n
if nargin <= 2  % nargin用来判断输入变量个数
    type = '01';
end;

if nargin > 1
    c = nclass;
    class_set = 1:c;
else
    class_set = unique(y);
    c = length(class_set);
end;

if strcmp(type, '01')
    Y = zeros(n, c);
    for cn = 1:c
        Y((y==class_set(cn)),cn) = 1;
        %y==class_set(cn) 表示标明行最大值所在列是class_set(cn)
        %的行位置，再将Y的这些行（cn列）的取值取1，即将原矩阵中每行最大值的位置取1，其余取0，构成Y
    end;
else
    Y = -1*ones(n, c);
    for cn = 1:c
        Y((y==class_set(cn)),cn) = 1;
    end;
end;