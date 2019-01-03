function [w,mbm_r,mbm_pdf] = mb2mbm01(mb_r,mb_pdf)
%MB2MBM01 converts a multi-Bernoulli process to a multi-Bernoulli mixture
%process with probability of existence r \in (0,1). 
%INPUT: mb_r: probability of existence --- vector of size (number of
%               Bernoulli components x 1).
%       mb_pdf: object density function if exists --- struct of size (number
%               of Bernoulli components x 1).
%OUTPUT:w: weights of multi-Bernoulli components in the multi-Bernoulli
%               mixture --- vector of size (number of multi-Bernoulli 
%               components). 
%       mbm_r: probability of existence --- cell of size (number of
%               multi-Bernoulli components x 1). Each cell contains a
%               vector of size (number of Bernoulli components x 1).
%       mbm_pdf: object density function if exists --- cell of size (number
%               of multi-Bernoulli components x 1). Each cell contains a 
%               struct of size (number of Bernoulli components x 1).

%NOTE: 
% 1. If a Bernoulli process has probability of existence r = 0, it is then 
% not represented in the multi-Bernoulli process. 
% 2. For the multi-Bernoulli included in the multi-Bernoulli mixture that has 
% no valid probability of existence, i.e., r = 0. The returned result should 
% satisfy that mbm_r{i} is a 0x1 empty double column vector, and that mbm_pdf{i}
% is a 0x1 empty struct array with the same fields as the input Bernoullis.
% 3. w sums up to one.

num_b = length(mb_r);

w = [];
mbm_r = [];
mbm_pdf = [];

for i = 1:num_b+1
    %Note that Bernoulli components in a multi-Bernoulli are unordered
    C = nchoosek(1:num_b,i-1);
    num = size(C,1);
    mbm_r_temp = cell(num,1);
    mbm_pdf_temp = cell(num,1);
    w_temp = zeros(num,1);
    for j = 1:num
        mbm_r_temp{j} = mb_r(C(j,:));
        mbm_pdf_temp{j} = mb_pdf(C(j,:));
        %Calculate weights of multi-Bernoulli components
        w_temp(j) = prod(1-mb_r)*prod(mb_r(C(j,:)))/prod(1-mb_r(C(j,:)));
    end
    %Append results of each combinatorics
    mbm_r = [mbm_r;mbm_r_temp];
    mbm_pdf = [mbm_pdf;mbm_pdf_temp];
    w = [w;w_temp];
end

%Normalise weights
w = w/sum(w);

end