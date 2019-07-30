% function s = log_cost(naturalLr,naturalPrior,targetOrNonTarget);
% function s = log_cost(naturalLr,naturalPrior,targetOrNonTarget,plotOrNot);
% 
% log_cost:  Computes the logarithmic cost score for a single posterior probability
%
% Lr values and priors are assumed to be referred to the target hypothesis
% 
% All probabilistic values are set in natural (not logarithmic) magnitudes
%
% Input parameters:
%      naturalLr:                  likelihood ratio in favor of the target hypothesis
%      naturalPrior:              prior probability f the target hypothesis
%      targetOrNonTarget:   0 if the truehypothesis is target, 1 otherwise
%      plot:                          0 if a plot is output, 1 otherwise. Default value: 0
%
% Author: Daniel Ramos-Castro, May 2007


function s = log_cost(naturalLr,naturalPrior,targetOrNonTarget,plotOrNot);

if nargin<4,
    plotOrNot=0;
end;

if (naturalLr==0 | naturalPrior==0)
    posterior=0;
    if targetOrNonTarget ==0
        s=inf;
    else
        s=0;
    end;
    
elseif (naturalPrior==1)
    posterior=1;
    if targetOrNonTarget ==0
        s=0;
    else
        s=inf;
    end;
else
    posterior=sigmoid(log(naturalLr)+logit(naturalPrior));
    if targetOrNonTarget ==0
        s=-log(posterior);
    else
        s=-log(1-posterior);
    end;
end;

if plotOrNot==1,
    f=figure;
    hold on;
    x=[0.00001:0.00001:0.99999];
    
    if targetOrNonTarget==0,
        y=-log(x);
        y_ext=[inf y 0];
        plot(x,y,'b','LineWidth',3);
        legend('H_p true','Location','NorthEast');
    else
        y=-log(1-x);
        y_ext=[0 y inf];
        plot(x,y,'r','LineWidth',3);
        legend('H_d true','Location','NorthWest');
    end;
   
    ax=get(f,'CurrentAxes');
    set(ax,'FontSize',16);
    axis([0 1 0 4]);
%    if posterior==0
%        s=y_ext(1);
%     elseif posterior==1
%         s=y_ext(length(y_ext));
%     else
%         s=y(find(x<posterior,1,'last'));
%     end;
    title(['Coste: ' num2str(s)],'FontSize',16); 
    
    plot([posterior posterior],[0 max(y)],'k--','LineWidth',3);
    xlabel('Probabilidad a Posteriori en favor de H_p','FontSize',16);
    ylabel('Coste','FontSize',16);
    grid;
    
end;

