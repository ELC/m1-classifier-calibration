function [EER, Calidad]=umbral (VEC_USUARIOS, VEC_IMPOSTORES)
    
    Resolucion= 1000;
    
    maximo= max([max(VEC_USUARIOS) max(VEC_IMPOSTORES)]);
    minimo= min([min(VEC_USUARIOS) min(VEC_IMPOSTORES)]);
    
    VEC_USUARIOS=round(Resolucion.*((VEC_USUARIOS-minimo)./(maximo-minimo)));
    VEC_IMPOSTORES=round(Resolucion.*((VEC_IMPOSTORES-minimo)./(maximo-minimo)));
    
    distr_usuarios=zeros(Resolucion+1,1);
    distr_impostores=zeros(Resolucion+1,1);
    densi_usuarios=zeros(Resolucion+1,1);
    densi_impostores=zeros(Resolucion+1,1);
    
    for i=1:length(VEC_USUARIOS)
        distr_usuarios(VEC_USUARIOS(i)+1)=distr_usuarios(VEC_USUARIOS(i)+1)+1;
    end
    
    for i=1:length(VEC_IMPOSTORES)
        distr_impostores(VEC_IMPOSTORES(i)+1)=distr_impostores(VEC_IMPOSTORES(i)+1)+1;
    end

    for i=1:Resolucion+1
        densi_usuarios(i)=sum(distr_usuarios(1:i));
        densi_impostores(i)=sum(distr_impostores(1:i));
    end
    densi_usuarios=densi_usuarios./sum(distr_usuarios);
    densi_impostores=1-densi_impostores./sum(distr_impostores);
    
    %plot([1:Resolucion+1],densi_usuarios,[1:Resolucion+1],densi_impostores)
    
    Calidad=min(max(densi_usuarios,densi_impostores));
    aux=find(max(densi_usuarios,densi_impostores)==Calidad);
    EER=sum(aux)/length(aux);
    EER=(maximo-minimo)*EER/Resolucion+minimo;
    Calidad=100-100*Calidad;
    