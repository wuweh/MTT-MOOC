function [w_miss] = missDetectHypothesis(P_D, P_G)
%MISSDETECTHYPOTHESIS calculates the likelihood of missed detection
%INPUT: P_D: detection probability
%       P_G: gating size in percentage
%OUTPUT: w_miss: missed detection hypothesis weight in logarithm domain --- scalar
w_miss = log(1-P_D*P_G);

end