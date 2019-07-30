function tippett_curso_cal(LR_Hp,LR_Hd,color)
% function tippett_curso_cal(LR_Hp,LR_Hd,color)

if nargin<3,
    color='b';
end;
    
rango=3;
resolucion=1000;
tipo=color;

dibuja_tippet_vecs(LR_Hp,LR_Hd,rango,resolucion,tipo);

[t_cal,nt_cal]=opt_loglr(log(LR_Hp),log(LR_Hd));

dibuja_tippet_vecs(exp(t_cal),exp(nt_cal),rango,resolucion,'k');
