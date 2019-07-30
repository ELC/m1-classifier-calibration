% function [targetLrs,nonTargetLrs]=calibracion_PAV_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest)
% function [targetLrs,nonTargetLrs]=calibracion_PAV_curso(scoresTargetTrain,scoresNonTargetTrain,scoresTargetTest,scoresNonTargetTest,prior)
%
% Calibrates using PAV.
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
function [llrs,mappingX,mappingY]=calibration_PAV(scoresTargetTrain,scoresNonTargetTrain,scoresTest,method)

if nargin<4
    method='raw';
end;

[llrTargetTrain,llrNonTargetTrain]=opt_loglr(scoresTargetTrain,scoresNonTargetTrain);

mappingX=([scoresTargetTrain scoresNonTargetTrain]);
mappingY=([llrTargetTrain llrNonTargetTrain]);

[mappingX,perturb]=sort(mappingX);
mappingY=mappingY(perturb);

if strcmp(method,'raw')
    % Elimination of same values of the mapping. Required by interp1.
    [mappingX,nonRepeatedValuesIndex]= unique(mappingX);
    mappingY=mappingY(nonRepeatedValuesIndex);

    % Because of the opt_llr function, the score-llr mapping will
    % have a very small slope in order to guarantee the idempotence of PAV
    % (appliying PAV to the output of PAV will give the same solution).
    % The output llrs will then need to have that small slope also in order to
    % preserve that idempotence. That is the reason of a linear
    % interpolation.
    llrs=interp1(mappingX,mappingY,scoresTest,'linear','extrap');
else
    % llrs=scoreToLlrPAVMappingCentroidInterpolation(scoresTest,mappingX,mappingY,method);
    % Interpolation of the centroids of the bins.
    peakThreshold=0.001;

    % Centroid interpolation of the PAV mapping
    % We use peak detection to find the centroids
    yLlrsMappingDifferences=diff(mappingY);

    peakIndexes=[1 find(yLlrsMappingDifferences > peakThreshold) length(mappingX)];

    xScoresMappingCentroids=zeros(1,length(peakIndexes)+1);
    xScoresMappingCentroids(1)=mappingX(1);
    xScoresMappingCentroids(length(xScoresMappingCentroids))=mappingX(length(mappingX));

    yLlrsMappingCentroids=zeros(1,length(peakIndexes)+1);
    yLlrsMappingCentroids(length(yLlrsMappingCentroids))=mappingY(length(mappingY));
    yLlrsMappingCentroids(length(yLlrsMappingCentroids)-1)=mappingY(length(mappingY)-1);

    for (countPeaks=2:length(peakIndexes)),
        xScoresMappingCentroids(countPeaks)=(mappingX(peakIndexes(countPeaks))+mappingX(peakIndexes(countPeaks-1)))/2;
        yLlrsMappingCentroids(countPeaks-1)=mappingY(peakIndexes(countPeaks-1));
    end;

    llrs=interp1(xScoresMappingCentroids,yLlrsMappingCentroids,scoresTest,method,'extrap');
end;

% figure;
% plot(sort(scoresTest),sort(llrs),'k');
% 
% title('Score to Log-LR mapping');
% xlabel('Scores');
% ylabel('Log-LR');