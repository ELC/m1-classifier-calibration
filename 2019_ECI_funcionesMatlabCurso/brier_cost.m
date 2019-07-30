% function s = brier_cost(naturalLr,naturalPrior,targetOrNonTarget);
% function s = brier_cost(naturalLr,naturalPrior,targetOrNonTarget,plotOrNot);
% 
%brier_cost:  Computes the brier score for a single posterior probability
%
% llr values and priors are assumed to be referred to the target hypothesis
%
% Input parameters:
%      llr:                             log-likelihood ratio in favor of the target hypothesis
%      prior:                         prior probability f the target hypothesis
%      targetOrNonTarget:   0 if the truehypothesis is target, 1 otherwise
%      plot:                          0 if a plot is output, 1 otherwise. Default value: 0
%
% Author: Daniel Ramos-Castro, May 2007

function s = brier_cost(naturalLr,naturalPrior,targetOrNonTarget,plotOrNot);

if nargin<4,
    plotOrNot=0;
end;

if (naturalLr==0 | naturalPrior==0)
    posterior=0;
    if targetOrNonTarget ==0
        s=1;
    else
        s=0;
    end;
    
elseif (naturalPrior==1)
    posterior=1;
    if targetOrNonTarget ==0
        s=0;
    else
        s=1;
    end;
else
    posterior=sigmoid(log(naturalLr)+logit(naturalPrior));
    if targetOrNonTarget ==0
        s=(1-posterior)^2;
    else
        s=(posterior)^2;
    end;
end;

if plotOrNot==1,
    f=figure;
    hold on;
    x=[0:0.00001:1];
    
    if targetOrNonTarget==0,
        y=(1-x).^2;
        plot(x,y,'b','LineWidth',3);
        legend('H_p true','Location','NorthEast');
    else
        y=(x).^2;
        plot(x,y,'r','LineWidth',3);
        legend('H_d true','Location','NorthWest');
    end;
   
    ax=get(f,'CurrentAxes');
    set(ax,'FontSize',16);
    axis([0 1 0 1]);

    title(['Coste: ' num2str(s)],'FontSize',16); 
    
    plot([posterior posterior],[0 max(y)],'k--','LineWidth',3);
    xlabel('Probabilidad a Posteriori en favor de H_p','FontSize',16);
    ylabel('Coste','FontSize',16);
    grid;
    
end;

