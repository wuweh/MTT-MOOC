function [miss_detect_likelihood, x, P] = missDetectHypothesis(x, P, P_D, P_G)
%MISSDETECTHYPOTHESIS calculates the likelihood of missed detection
%INPUT: P_D: detection probability
%       P_G: gating size in percentage
%OUTPUT: miss_detect_likelihood: likelihood of missed detection
miss_detect_likelihood = 1-P_D*P_G;

end

