function miss_detect_likelihood = missDetectLikelihood(P_D,P_G)
%MISSDETECTLIKELIHOOD calculates the likelihood of missed detection
%INPUT: P_D: detection probability
%       P_G: gating size in percentage
%OUTPUT: miss_detect_likelihood: likelihood of missed detection
miss_detect_likelihood = 1-P_D*P_G;

end

