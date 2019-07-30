% function [targetLrs,nonTargetLrs]=calibracion_lineal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest)
% function [targetLrs,nonTargetLrs]=calibracion_lineal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest,prior)
%
% Calibrates using linear logistic regression.
%
% -- INPUT --
% * scoresTargetTrain:    (1xNt) Target scores used for calibration training (natural scale).
% * scoresNonTargetTrain: (1xNn) Non-target scores used for calibration training (natural scale).
% * scoresTargetTest:           (1xN)  Scores target to be calibrated (matural scale).
% * scoresNonTargetTest:           (1xN)  Scores nonTarget to be calibrated (natural scale).
% * prior:                Prior probability for the logistic regression
%                         objective. Default value: 0.5
%
% -- OUTPUT --
% * llrs:                 (1xN)  Output calibrated llrs.
%
% This software makes use of the FoCal toolkit
% (see http://www.dsp.sun.ac.za/~nbrummer/focal)
%
% Author: Daniel Ramos, November 2006.
function  [targetLrs,nonTargetLrs]=calibracion_lineal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest)

if nargin<5
    prior=0.5;
end;

% It has been detected that in some cases the convergence of the FoCal CG
% algorithm diverges or is not achieving a good optimum. We use a
% provisional conservative training strategy, which is not intended to be a
% definitive solution. 
disp('Training with a conservative threshold...');
conservativeThreshold=1e-10;
[w,minimumConvergence]=train_llr_fusion_convergence_threshold(log(scoresTargetTrain),log(scoresNonTargetTrain),conservativeThreshold,prior);
if (minimumConvergence>=conservativeThreshold)
    disp(' ');
    disp('WARNING: Convergence not achieved with a conservative threshold.');
    disp('  Check the convergence of the algorithm (see "a posteriori" APE plot).');
    disp(' ');
    disp('Training with the minimum convergence value achieved in the previous step...');
    w=train_llr_fusion_convergence_threshold(log(scoresTargetTrain),log(scoresNonTargetTrain),conservativeThreshold,prior);
else
    disp('Done!');
end;

targetLrs=exp(lin_fusion(w,log(scoresTargetTest)));
nonTargetLrs=exp(lin_fusion(w,log(scoresNonTargetTest)));

h=figure;
% Sorting scores and LRs separately is OK because the trasformation is monotonic
subplot(2,1,1);

plot(sort([scoresTargetTest scoresNonTargetTest]),sort([targetLrs nonTargetLrs]),'b');
xlabel('LR de entrada','FontSize',16);
ylabel('LR de salida','FontSize',16);
legend('Escala Natural','Location','SouthEast')

ah=get(h,'CurrentAxes');
set(ah,'FontSize',16);

subplot(2,1,2);

plot(sort(log10([scoresTargetTest scoresNonTargetTest])),sort(log10([targetLrs nonTargetLrs])),'r');
xlabel('log10(LR) de entrada','FontSize',16);
ylabel('log10(LR) de salida','FontSize',16);
legend('Escala Log10','Location','SouthEast')

ah=get(h,'CurrentAxes');
set(ah,'FontSize',16);