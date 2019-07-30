% function [targetLrs,nonTargetLrs]=calibracion_scal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest)
% function [targetLrs,nonTargetLrs]=calibracion_scal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest,prior)
%
% Calibrates using s-cal.
%
% -- INPUT --
% * scoresTargetTrain:    (1xNt) Target scores used for calibration training (natural scale).
% * scoresNonTargetTrain: (1xNn) Non-target scores used for calibration training (natural scale).
% * scoresTargetTest:           (1xN)  Scores target to be calibrated (matural scale).
% * scoresNonTargetTest:           (1xN)  Scores nonTarget to be calibrated (natural scale).
%
% -- OUTPUT --
% * llrs:                 (1xN)  Output calibrated llrs.
%
% This software makes use of the FoCal toolkit
% (see http://www.dsp.sun.ac.za/~nbrummer/focal)
%
% Author: Daniel Ramos, November 2006.
function  [targetLrs,nonTargetLrs]=calibracion_scal_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest)

if nargin<5
    prior=0.5;
end;

targetLrs=exp(calibration_scal(log(scoresTargetTrain),log(scoresNonTargetTrain),log(scoresTargetTest)));
nonTargetLrs=exp(calibration_scal(log(scoresTargetTrain),log(scoresNonTargetTrain),log(scoresNonTargetTest)));

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