% function histograma_curso(LR_Hp,LR_Hd);
% 
% Pinta histogramas. Para prácticas de cursos.
%
% Entrada:
%   LR_Hp:  Valores de LR para los que la hipótesis del fiscal es cierta
%   LR_Hd:  Valores de LR para los que la hipótesis de la defensa es cierta
%
% Author: Daniel Ramos-Castro, May 2007
function histograma_curso(LR_Hp,LR_Hd);

h=figure;

subplot(2,1,1);
count=hist(log(LR_Hp),50);
hist(log(LR_Hp),50);
legend('H_p cierta');

axis([log(min([LR_Hp LR_Hd])) log(max([LR_Hp LR_Hd])) 0 max(count)]);

subplot(2,1,2);
count=hist(log(LR_Hd),50);
hist(log(LR_Hd),50);
legend('H_d cierta');

axis([log(min([LR_Hp LR_Hd])) log(max([LR_Hp LR_Hd])) 0 max(count)])