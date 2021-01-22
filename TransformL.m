function Y = TransformL(y, nclass, type)
% [~, g] = max(F,[],2); % g��n*1����ÿ��ֵΪFÿ�����ֵ���±�λ��
% Gsr4 = TransformL(g, class_num);
% TransformL()��ԭ����F��ÿ�����ֵ��λ��ȡ1������ȡ0������Y
% �������ı�ʶ����F һ�������ֵ��λ�� ת��Ϊ 0-1 ��ʶ����Y
% y��һ����������yi��ԭ����ÿ�����ֵ�����±�

% �ȼ��ڣ�������Y��ÿ�����ֵ��λ��ȡ1������ȡ0
% for i = 1:n
%     [~,mix] = max(Y(i,:));
%     Y(i,:) = 0;
%     Y(i,mix) = 1;
% end



n =length(y);  %ԭ�����n
if nargin <= 2  % nargin�����ж������������
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
        %y==class_set(cn) ��ʾ���������ֵ��������class_set(cn)
        %����λ�ã��ٽ�Y����Щ�У�cn�У���ȡֵȡ1������ԭ������ÿ�����ֵ��λ��ȡ1������ȡ0������Y
    end;
else
    Y = -1*ones(n, c);
    for cn = 1:c
        Y((y==class_set(cn)),cn) = 1;
    end;
end;