function [w_miss, x, P] = missDetectHypothesis(x, P, P_D, P_G)
%MISSDETECTHYPOTHESIS calculates the likelihood of missed detection
%INPUT: P_D: detection probability
%       P_G: gating size in percentage
%OUTPUT: w_miss: missed detection hypothesis weight --- scalar
w_miss = 1-P_D*P_G;

end