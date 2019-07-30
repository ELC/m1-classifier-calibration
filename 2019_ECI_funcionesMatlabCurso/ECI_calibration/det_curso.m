function det_curso(LR_Hp,LR_Hd,color)
% function det_curso(LR_Hp,LR_Hd,color)

if nargin<3,
    color='b';
end;

dibujadet_dramos(LR_Hp,LR_Hd,color,2);
legend('');