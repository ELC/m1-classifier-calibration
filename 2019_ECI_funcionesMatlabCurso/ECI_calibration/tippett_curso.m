function tippett_curso(LR_Hp,LR_Hd,color)
% function tippett_curso(LR_Hp,LR_Hd,color)

if nargin<3,
    color='b';
end;
    
rango=3;
resolucion=1000;
tipo=color;

dibuja_tippet_vecs(LR_Hp,LR_Hd,rango,resolucion,tipo);
