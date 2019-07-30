% function
% llrs=calibration_sCal(scoresTargetTrain,scoresNonTargetTrain,scoresTest)
%
% Calibrates using FoCal toolkit's sCal.
%
% -- INPUT --
% * scoresTargetTrain:    (1xNt) Target scores used for calibration training.
% * scoresNonTargetTrain: (1xNn) Non-target scores used for calibration training.
% * scoresTest:           (1xN)  Scores to be calibrated.
%
% -- OUTPUT --
% * llrs:                 (1xN)  Output calibrated llrs.
%
% This software makes use of the FoCal toolkit
% (see http://www.dsp.sun.ac.za/~nbrummer/focal)
%
% Author: Daniel Ramos, November 2006.
function llrs=calibration_sCal(scoresTargetTrain,scoresNonTargetTrain,scoresTest)

if nargin<4
    prior=0.5;
end;

[a,b,alpha,beta]=train_s_cal(scoresTargetTrain,scoresNonTargetTrain);
llrs=s_cal(a,b,alpha,beta,scoresTest);

% figure;
% plot(sort(scoresTest),sort(llrs),'k');
% 
% title('Score to Log-LR mapping');
% xlabel('Scores');
% ylabel('Log-LR');