function [Y,changed,it] = main2_updateY(F,Q,Y)
changed = 0;
[n,c] = size(Y);
G = F*Q;
yg = diag(Y'*G)';
yy = diag(Y'*Y+ eps*eye(c))';
for it = 1:10
    converged = true;
    for i = 1:n
        yi = Y(i,:);
        gi = G(i,:);
        [~,id0] = max(yi);
        ss = (yg + gi.*(1-yi))./sqrt(yy + 1-yi) - (yg - gi.*yi)./sqrt(yy - yi);
        [~,id] = max(ss(:));
        if id ~= id0
            converged = false;
            changed = changed + 1;
           %% update Y
            yi = zeros(1,c);
            yi(id) = 1;
            Y(i,:) = yi;
            %% update yy
            yy(id0) = yy(id0) - 1;
            yy(id) = yy(id) + 1;
           
            %% update yg
            yg(id0) = yg(id0) - gi(id0);
            yg(id) = yg(id) + gi(id);
        end
    end
    if converged
        break;
    end
end