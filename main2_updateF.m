function [F,t,obj_gpi] = main2_updateF(A_tail,Y,c,F,Q,parameter)

M_temp = (Y'*Y+eps*eye(c))^(-0.5);
M = Y*M_temp;
B = M*Q'; 
err_mean=1;
t = 1;
while err_mean>1e-4
    M_F = 2*A_tail*F + 2*parameter*B;
    [U_F,~,V_F] = svd(M_F,'econ');
    F = U_F*V_F';
    clear U_F V_F;
    obj_gpi(t) = trace(F'*A_tail*F)+2*parameter*trace(F'*B);   
    if t>=3
        mean1 = mean(obj_gpi(t-2)+obj_gpi(t-1));
        mean2 = mean(obj_gpi(t-1)+obj_gpi(t));
        err_mean = abs(mean2 - mean1);
    end
    t = t+1;
    if t > 5000
        break;
    end
end

