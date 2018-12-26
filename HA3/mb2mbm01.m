function [w,mbm_r,mbm_pdf] = mb2mbm01(mb_r,mb_pdf)
%MB2MBM01 converts a multi-Bernoulli process to a multi-Bernoulli mixture
%process with probability of existence r \in [0,1]. 
%INPUT: mb_r: probability of existence --- vector of size (number of
%               Bernoulli components x 1).
%       mb_pdf: object density function if exists --- struct of size (number
%               of Bernoulli components x 1).
%OUTPUT:w: weights of multi-Bernoulli components in the multi-Bernoulli
%               mixture --- vector of size (number of multi-Bernoulli 
%               components). (SUM UP TO ONE).
%       mbm_r: probability of existence --- cell of size (number of
%               multi-Bernoulli components x 1). Each cell contains a
%               vector of size (number of Bernoulli components x 1).
%       mbm_pdf: object density function if exists --- cell of size (number
%               of multi-Bernoulli components x 1). Each cell contains a 
%               struct of size (number of Bernoulli components x 1).

%NOTE: if a Bernoulli process has probability of existence r = 0, it is
%then not represented in the multi-Bernoulli process.

idx1 = mb_r == 1;
mb_r1 = mb_r(idx1);
mb_pdf_r1 = mb_pdf(idx1);

idx2 = mb_r < 1;
mb_r = mb_r(idx2);
mb_pdf = mb_pdf(idx2);

w = [];
mbm_r = [];
mbm_pdf = [];

num_mb = length(mb_r);
for i = 1:num_mb
    C = nchoosek(1:num_mb,i);
    num = size(C,1);
    mbm_r_temp = cell(num,1);
    mbm_pdf_temp = cell(num,1);
    w_temp = zeros(num,1);
    for j = 1:num
        mbm_r_temp{j} = [mb_r1;mb_r(C(j,:))];
        mbm_pdf_temp{j} = [mb_pdf_r1;mb_pdf(C(j,:))];
        w_temp(j) = prod(1-mb_r)*prod(mb_r(C(j,:)))/prod(1-mb_r(C(j,:)));
    end
    mbm_r = [mbm_r;mbm_r_temp];
    mbm_pdf = [mbm_pdf;mbm_pdf_temp];
    w = [w;w_temp];
end

w = w/sum(w);

end
