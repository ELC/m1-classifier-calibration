% [EER_DET]=getEER_DET(pFA,pFR)
% Saca el EER de la curva DET que generarían los vectores
% de FA y FR pFA y pFR
% Si hay error devuelve EER_DET=-1
function [EER_DET]=getEER_DET(pFA,pFR)

epsilon=1e-3;
hay_eer=0;
%Estimación del EER por curva DET
while(1),
   for i=1:length(pFA),
      if abs(pFA(i)-pFR(i))<epsilon
         EER_DET=pFA(i)*100;
      	% disp(sprintf('EER_DET: %0.4g%%',EER_DET));
      	hay_eer=1;
      	break;
   	end;
	end;   
   if hay_eer~=1
      epsilon=2*epsilon;
      if epsilon>1
         EER_DET=-1;
         % disp('¡NO ES POSIBLE ESTIMAR EER!');
         break;
      end;
   else
      break;
   end;
end;
