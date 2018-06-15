E1 = E1(:,:,[1:24,26:50]); E2 = E2(:,:,[1:24,26:50]); E3 = E3(:,:,[1:24,26:50]); E4 = E4(:,:,[1:24,26:50]);
GT = GT(:,:,[1:24,26:50]);
tr = 49; tes = 1; tee = 48;
siz = [544 544 49];
%%
Gd = Gd(:,:,[1:24,26:50]); Gold = Gold(:,:,[1:24,26:50]); HA = HA(:,:,[1:24,26:50]);
Iodine = Iodine(:,:,[1:24,26:50]); Lipid = Lipid(:,:,[1:24,26:50]); Water = Water(:,:,[1:24,26:50]);

%%
%background segmentation
bg = zeros(siz);
bg(E1 < 0.13) = 1;

%%
%train test
siztr = [siz(1),siz(2),1];
sizte = [siz(1),siz(2),(tee-tes+1)];
masktr = not(bg(:,:,tr));
maskte = not(bg(:,:,tes:tee));

%train test
E1tr = E1(:,:,tr); E2tr = E2(:,:,tr); E3tr = E3(:,:,tr); E4tr = E4(:,:,tr); GTtr = GT(:,:,tr);
E1te = E1(:,:,tes:tee); E2te = E2(:,:,tes:tee); E3te = E3(:,:,tes:tee); E4te = E4(:,:,tes:tee); GTte = GT(:,:,tes:tee);
E1train = zeros(siztr); E1train(masktr) = E1tr(masktr);
E2train = zeros(siztr); E2train(masktr) = E2tr(masktr); 
E3train = zeros(siztr); E3train(masktr) = E3tr(masktr); 
E4train = zeros(siztr); E4train(masktr) = E4tr(masktr); 
GTtrain = zeros(siztr); GTtrain(masktr) = GTtr(masktr); 

E1test = zeros(sizte); E1test(maskte) = E1te(maskte); 
E2test = zeros(sizte); E2test(maskte) = E2te(maskte); 
E3test = zeros(sizte); E3test(maskte) = E3te(maskte); 
E4test = zeros(sizte); E4test(maskte) = E4te(maskte); 
GTtest = zeros(sizte); GTtest(maskte) = GTte(maskte); 
%%
%train test material
HAtr = HA(:,:,tr); Goldtr = Gold(:,:,tr); Gdtr = Gd(:,:,tr); Iodinetr = Iodine(:,:,tr); Lipidtr = Lipid(:,:,tr); Watertr = Water(:,:,tr);
HAte = HA(:,:,tes:tee); Goldte = Gold(:,:,tes:tee); Gdte = Gd(:,:,tes:tee); Iodinete = Iodine(:,:,tes:tee); Lipidte = Lipid(:,:,tes:tee); Waterte = Water(:,:,tes:tee); 
HAtrain = zeros(siztr);     HAtrain(masktr) = HAtr(masktr);
Goldtrain = zeros(siztr);   Goldtrain(masktr) = Goldtr(masktr); 
Gdtrain = zeros(siztr);     Gdtrain(masktr) = Gdtr(masktr); 
Iodinetrain = zeros(siztr); Iodinetrain(masktr) = Iodinetr(masktr); 
Lipidtrain = zeros(siztr);  Lipidtrain(masktr) = Lipidtr(masktr); 
Watertrain = zeros(siztr);  Watertrain(masktr) = Watertr(masktr); 

HAtest = zeros(sizte);     HAtest(maskte) = HAte(maskte);
Goldtest = zeros(sizte);   Goldtest(maskte) = Goldte(maskte); 
Gdteest = zeros(sizte);     Gdtest(maskte) = Gdte(maskte); 
Iodinetest = zeros(sizte); Iodinetest(maskte) = Iodinete(maskte); 
Lipidtest = zeros(sizte);  Lipidtest(maskte) = Lipidte(maskte); 
Watertest = zeros(sizte);  Watertest(maskte) = Waterte(maskte); 
%%
Xtr = [E1train(masktr) E2train(masktr) E3train(masktr) E4train(masktr)];
Xte = [E1test(maskte) E2test(maskte) E3test(maskte) E4test(maskte)];
%Xtr = E1train(masktr);
%Xte = E1test(maskte);

clearvars E1tr E2tr E3tr E4tr E1te E2te E3te E4te
%%
%atlas
sig = 1.25000001; 
K = 13;
atlasmat = cell(K-1,1);
atlastestmat = cell(sizte(3),K-1);
for k = 1:K-1
    atlastemp = zeros(siztr); atlastemp(GTtrain == k) = 1;
    atlasmat{k,1} = imgaussfilt3(atlastemp,sig);
    %atlasmat{k,1} = atlastemp;
    atlastemp = cell2mat(atlasmat(k,1));
    for n = 1:sizte(3)
        atlastestmat{n,k} = atlastemp(maskte(:,:,n));
    end
end

%atlas test
atlasbase = zeros(sizte); atlasbase(maskte) =1;
for n = 1:sizte(3)
    for k =1:K-1
        atlasbase(:,:,n) = atlasbase(:,:,n) - cell2mat(atlasmat(k,1));
    end
end

atlas = zeros(sum(maskte(:) ==1),K);
atlas(:,1:12) = cell2mat(atlastestmat);
atlas(:,K) = atlasbase(maskte);
%%
%
test = zeros(sizte);
test(maskte) = atlasbase(maskte);
figure;
imagesc(test(:,:,1)');
axis tight equal off
%%
%EM initialvalue
%material
GTmat = GTtrain(masktr);
for k = 1:K-1
    tmp1 = Xtr(:,1); tmp2 = Xtr(:,2); tmp3 = Xtr(:,3); tmp4 = Xtr(:,4);
    S.mu(k,1) = mean(tmp1(GTmat == k));
    S.mu(k,2) = mean(tmp2(GTmat == k));
    S.mu(k,3) = mean(tmp3(GTmat == k));
    S.mu(k,4) = mean(tmp4(GTmat == k));
    S.Sigma(:,:,k) = cov([tmp1(GTmat == k),tmp2(GTmat == k),tmp3(GTmat == k),tmp4(GTmat == k)]);
    S.ComPro(k,1) = numel(tmp1(GTmat == k));
end

%base
mask = or(bg(:,:,tr),logical(GTtrain));
mask = not(mask);
base = zeros(siztr);
base(mask) = E1train(mask);

%base
maskbase = logical(base); 
S.mu(K,1) = mean(E1train(maskbase));
S.mu(K,2) = mean(E2train(maskbase));
S.mu(K,3) = mean(E3train(maskbase));
S.mu(K,4) = mean(E4train(maskbase));
S.Sigma(:,:,K) = cov([E1train(maskbase),E2train(maskbase),E3train(maskbase),E4train(maskbase)]);
S.ComPro(K,1) = numel(E1train(maskbase));
S.ComPro = S.ComPro ./ sum(S.ComPro);
clearvars tmp1 tmp2 tmp3 tmp4
%%
[Imap,L,PP,GMMMu,GMMSigma,GMMpro] = AtlasGuidedEM_kubo(Xte,atlas,S,K,maskte,sizte);
%%
%EM initialvalue
%material
GTmat = GTtrain(masktr);
for k = 1:K-1
    tmp1 = Xtr(:,1);
    S.Mu(k,1) = mean(tmp1(GTmat == k));
    S.Sigma(:,:,k) = cov(tmp1(GTmat == k));
    S.ComPro(k,1) = numel(tmp1(GTmat == k));
end

%base
mask = or(bg(:,:,tr),logical(GTtrain));
mask = not(mask);
base = zeros(siztr);
base(mask) = E1train(mask);

%base
maskbase = logical(base); 
S.Mu(K,1) = mean(E1train(maskbase));
S.Sigma(:,:,K) = cov(E1train(maskbase));
S.ComPro(K,1) = numel(E1train(maskbase));
S.ComPro = S.ComPro ./ sum(S.ComPro);
clearvars tmp1
%%
%EM(With atlas kubo)
K = 13; %class
F = size(Xte,2); %feature
N = size(Xte,1); %sample
log_2pi = log(power(2*pi, F/2));

%initial value
GMMMu = S.Mu;
GMMSigma = S.Sigma;
GMMpro = atlas;
nMaxIteration = 100;
tolerance = 1e-4;

for nIteration = 0:nMaxIteration
    %disp(nIteration);
    
    %Weight
    p = zeros(N,K);
    for k = 1:K
        p(:,k) = mvnpdf(Xte,GMMMu(k,:),GMMSigma(:,:,k)); 
    end
    p = p.*GMMpro;
    pp = p ./ sum(p,2);

    tmp = sum(p,2);
    r = tmp>0;
    fLikelihood = sum(log(tmp(r))-log_2pi);
    
    if( nIteration ~= 0)
        if( fLastLikelihood >= fLikelihood || 2.0 * abs( fLastLikelihood - fLikelihood ) < tolerance * (abs( fLastLikelihood ) + abs( fLikelihood ) ) )
            if(fLastLikelihood >= fLikelihood)
                GMMMu = GMMMuOld;
                GMMSigma = GMMSigmaOld;
            else
                GMMpro = pp;
            end
            break;
        end
    elseif nIteration == nMaxIteration
        break;
    end
    GMMpro = pp;
    
    % 前回（今回）の尤度和を保存
    fLastLikelihood = fLikelihood;
    disp(fLikelihood);
    GMMMuOld = GMMMu;
    GMMSigmaOld = GMMSigma;
       
    %proportion
    nk = sum(pp,1);
    GMMpro = pp;
    
    %mean
    tmp = zeros(N,F);
    for k = 1:K
        for f = 1:F
            tmp(:,f) = pp(:,k) .* Xte(:,f);
        end
        GMMMu(k,:) = nansum(tmp,1)./nk(k);
    end

    %sigma
    for k = 1:K
        Xmu = (Xte - GMMMu(k,:));
        pp = reshape(pp,[N,1,K]);
        GMMSigma(:,:,k) = (pp(:,1,k) .* Xmu)' * Xmu;
        GMMSigma(:,:,k) = GMMSigma(:,:,k) ./ nk(1,k);
    end
    pp = reshape(pp,[N K]);
    
end
%%
%MAP(With atlas)
y = cell(1,K);
for k = 1:K
    y{1,k} = GMMpro(:,k) .* mvnpdf(Xte,GMMMu(k,:),GMMSigma(:,:,k));
end
Feat = cell2mat(y);
p_l = atlas;
PP = bsxfun(@times,Feat,p_l);
p_x = sum(PP,2);
PP = bsxfun(@rdivide,PP,p_x);
[~,L] = max(PP,[],2);

%%
%Colormap
map = [0, 0, 0
    parula(12), 
    0.5, 0.5, 0.5];
imagesc(GTtrain');
colormap(map);
axis tight equal
caxis([0 13]);
%%
%figure
slicenum = 1;

I = zeros(sizte);
I(maskte) = L;
%figure;
label = I(:,:,slicenum);
imagesc(label'); 
axis equal tight off
colormap(map);
caxis([0 13]);
%%

PP1 = zeros(sizte);
PP1(maskte) = PP(:,1);
figure;
subplot(4,4,1)
imagesc(PP1(:,:,slicenum)');
axis equal tight 
colormap hot

PP2 = zeros(sizte); 
PP2(maskte) = PP(:,2);
subplot(4,4,2)
imagesc(PP2(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP3 = zeros(sizte);
PP3(maskte) = PP(:,3);
subplot(4,4,3)
imagesc(PP3(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP4 = zeros(sizte);
PP4(maskte) = PP(:,4);
subplot(4,4,4)
imagesc(PP4(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP5 = zeros(sizte);
PP5(maskte) = PP(:,5);
subplot(4,4,5)
imagesc(PP5(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP6 = zeros(sizte);
PP6(maskte) = PP(:,6);
subplot(4,4,6)
imagesc(PP6(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP7 = zeros(sizte); 
PP7(maskte) = PP(:,7);
subplot(4,4,7)
imagesc(PP7(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP8 = zeros(sizte);
PP8(maskte) = PP(:,8);
subplot(4,4,8)
imagesc(PP8(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP9 = zeros(sizte);
PP9(maskte) = PP(:,9);
subplot(4,4,9)
imagesc(PP9(:,:,slicenum)');
axis equal tight 
colormap hot

PP10 = zeros(sizte);
PP10(maskte) = PP(:,10);
subplot(4,4,10)
imagesc(PP10(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP11 = zeros(sizte);
PP11(maskte) = PP(:,11);
subplot(4,4,11)
imagesc(PP11(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP12 = zeros(sizte);
PP12(maskte) = PP(:,12);
subplot(4,4,12)
imagesc(PP12(:,:,slicenum)'); 
axis equal tight 
colormap hot

PP13 = zeros(sizte);
PP13(maskte) = PP(:,13);
subplot(4,4,13)
imagesc(PP13(:,:,slicenum)'); 
axis equal tight 
colormap hot


%%
subplot(1,2,1)
imagesc(Output(:,:,13)');
axis tight equal off
caxis([0 13]);
colormap(map)

subplot(1,2,2)
imagesc(Output(:,:,37)');
axis tight equal off
caxis([0 13]);
colormap(map)

%%
%JI
JI = zeros(K-1,1);
for k = 1:K-1
A = Imap == k;
B = GTtest == k;
A = A(:);
B = B(:);
JI(k,1) =  sum(and(A,B)) ./ sum(or(A,B));
end
clearvars A B
disp(JI);

%%
RP = cell(1,K);
for k = 1:K
    RP{1,k} = -log(PP(:,k)+eps);
end
RP = cell2mat(RP);

[lambda,h] =ndgrid(0.005:0.001:0.01,1:0.1:3.0);
lambda = lambda(:);
h = h(:);

%%

h = 2.7;
lambda = 0.08;
sumJI = zeros(size(h,1),1);
%for n = 1:size(h,1)
%n=62;
GraphModel = CreateFullyConnectedGraphWithMask(maskte);
Sigmat =  abs(bsxfun(@minus,GMMMu(:,1),GMMMu(:,1)'))*h +eye(K);
Kmat = ones(K);

CurLabel = ones(N,1);
PreLabel = zeros(N,1);
Output = zeros(sizte);
flag = 0;
PreE =0;
%%
while(flag ~=1)
    for  k = 1:K
        PropLabel = zeros(N,1)+k;
        disp(k);
        GraphModel = SetTWeights(GraphModel,RP,CurLabel,PropLabel,lambda,N);
        GraphModel = SetNWeights(GraphModel,E1test(maskte),CurLabel,PropLabel,Sigmat,Kmat);
        [lowerBound, labels] = qpboMex([GraphModel.Vs,GraphModel.Vt],[GraphModel.Hi,GraphModel.Hj,GraphModel.H00,GraphModel.H01,GraphModel.H10,GraphModel.H11]);
        labels = logical(labels);
        CurLabel(labels) = PropLabel(labels);
        
        Eunary = sum(GraphModel.Vs);
        Epairwise = sum(GraphModel.H00);
        E = Eunary + Epairwise;
        %{
        Output(maskte) = CurLabel;
        imagesc(Output(:,:,1)'); 
        axis equal tight off
        colormap(map);
        %pause;
        %}
    end
    
    disp(E);
    if CurLabel == PreLabel
        flag = 1;
        Output(maskte) = CurLabel;
    end
    
    if E == PreE
        flag = 1;
        Output(maskte) = CurLabel;
    end
    
    PreLabel = CurLabel;
    PreE = E;
end

JI = zeros(K-1,1);
for k = 1:K-1
    A = Output == k;
    B = GTtest == k;
    A = A(:);
    B = B(:);
    JI(k,1) =  sum(and(A,B)) ./ sum(or(A,B));
end
clearvars A B
disp(mean(JI));
sumJI(n) = sum(JI);
disp(n);

%end


%%
subplot(1,2,1)
imagesc(Output(:,:,13)');
axis tight equal off
caxis([0 13]);
colormap(map)

subplot(1,2,2)
imagesc(Output(:,:,37)');
axis tight equal off
caxis([0 13]);
colormap(map)
%%

imagesc(I(:,:,17)');
axis tight equal off
caxis([0 13]);
colormap(map)




%%
%EM initialvalue
%material
GTmat = GTtest(maskte);
for k = 1:K-1
    tmp1 = Xte(:,1); tmp2 = Xte(:,2); tmp3 = Xte(:,3); tmp4 = Xte(:,4);
    ST.Mu(k,1) = mean(tmp1(GTmat == k));
    ST.Mu(k,2) = mean(tmp2(GTmat == k));
    ST.Mu(k,3) = mean(tmp3(GTmat == k));
    ST.Mu(k,4) = mean(tmp4(GTmat == k));
    ST.Sigma(:,:,k) = cov([tmp1(GTmat == k),tmp2(GTmat == k),tmp3(GTmat == k),tmp4(GTmat == k)]);
    ST.ComPro(k,1) = numel(tmp1(GTmat == k));
end
%%
%base
mask = or(bg(:,:,tes:tee),logical(GTtest));
mask = not(mask);
base = zeros(sizte);
base(mask) = E1test(mask);

%base
maskbase = logical(base); 
ST.Mu(K,1) = mean(E1test(maskbase));
ST.Mu(K,2) = mean(E2test(maskbase));
ST.Mu(K,3) = mean(E3test(maskbase));
ST.Mu(K,4) = mean(E4test(maskbase));
ST.Sigma(:,:,K) = cov([E1test(maskbase),E2test(maskbase),E3test(maskbase),E4test(maskbase)]);
ST.ComPro(K,1) = numel(E1test(maskbase));
ST.ComPro = ST.ComPro ./ sum(ST.ComPro);
clearvars tmp1 tmp2 tmp3 tmp4




%%
%EM(With atlas)
nMaxIteration =10;
tolerance = 1e-4;
[ave, sigma, Weight] = AtlasGuidedEM( Xte, S.Mu',S.Sigma, atlas, nMaxIteration, tolerance );

%MAP(With atlas)
y = cell(1,K);
for k = 1:K
    %y{1,k} = GMMpro(:,k) .* mvnpdf(In,GMMMu(k,:),GMMSigma(:,:,k));
    y{1,k} = Weight(:,1,k) .* mvnpdf(Xte,ave(1,:,k),sigma(:,:,k));
end
Feat = cell2mat(y);
p_l = atlas;
PP = bsxfun(@times,Feat,p_l);
p_x = sum(PP,2);
PP = bsxfun(@rdivide,PP,p_x);
[~,L] = max(PP,[],2);











%%
%EM(Without atlas)
%In = Xtr;
K = 13; %class
F = size(Xte,2); %feature
N = size(Xte,1); %sample
GMModel = fitgmdist(Xte,13,'start',S,'Options',statset('Display','iter','MaxIter',100));
GMMMu = GMModel.mu;
GMMSigma = GMModel.Sigma;
GMMpro = GMModel.ComponentProportion;

%MAP(Without atlas)
y = cell(1,K);
for k = 1:K
    y{1,k} = GMMpro(k) * mvnpdf(Xte,GMMMu(k,:),GMMSigma(:,:,k));
end
Feat = cell2mat(y);
p_l = GMMpro;
PP = bsxfun(@times,Feat,p_l);
p_x = sum(PP,2);
PP = bsxfun(@rdivide,PP,p_x);
[~,L] = max(PP,[],2);


%%
%{
%%
%atlas
sig = 5.0; w=17;
K = 13;
%atlas = zeros(sum(masktr(:) ==1),K);
atlasmat = cell(K,1); atlasbase = zeros(siztr); atlasbase(masktr) =1;
for k = 1:K-1
    atlastemp = zeros(siztr); atlastemp(GTtrain == k) = 1; 
    atlastemp = imgaussfilt3(atlastemp,sig,'FilterSize',w);
    atlasmat{k,1} = zeros(siztr); atlasmat{k,1}(masktr) = atlastemp(masktr);
    atlasbase = atlasbase - cell2mat(atlasmat(k,1));
    %atlas(:,k) = atlastemp(masktr);
end
atlasmat{K,1} = zeros(siztr);
atlasmat{K,1}(masktr) = atlasbase(masktr);
%atlas(:,K) = atlasbase(masktr);

%atlas test
atlas = zeros(sum(maskte(:) ==1),K);
atlastemp = zeros(sizte);
for k =1:K
    temp = cell2mat(atlasmat(k,1));
    temp2 = temp(maskte);
    %temp = repmat(temp,sizte(3),1);
   % atlas(:,k) = temp;
end


%}
%{
%%
%EM test 正解
%material
GTmat = GTtest(maskte);
for k = 1:K-1
    tmp1 = Xte(:,1); tmp2 = Xte(:,2); tmp3 = Xte(:,3); tmp4 = Xte(:,4);
    STest.mu(k,1) = mean(tmp1(GTmat == k));
    STest.mu(k,2) = mean(tmp2(GTmat == k));
    STest.mu(k,3) = mean(tmp3(GTmat == k));
    STest.mu(k,4) = mean(tmp4(GTmat == k));
    STest.Sigma(:,:,k) = cov([tmp1(GTmat == k),tmp2(GTmat == k),tmp3(GTmat == k),tmp4(GTmat == k)]);
    STest.ComPro(k,1) = numel(tmp1(GTmat == k));
end

%base
mask = or(bg(:,:,tes:tee),logical(GTtest));
mask = not(mask);
base = zeros(sizte);
base(mask) = E1test(mask);
maskbase = logical(base); 

STest.mu(K,1) = mean(E1test(maskbase));
STest.mu(K,2) = mean(E2test(maskbase));
STest.mu(K,3) = mean(E3test(maskbase));
STest.mu(K,4) = mean(E4test(maskbase));
STest.Sigma(:,:,K) = cov([E1test(maskbase),E2test(maskbase),E3test(maskbase),E4test(maskbase)]);
STest.ComPro(K,1) = numel(E1test(maskbase));
STest.ComPro = STest.ComPro ./ sum(STest.ComPro);
clearvars tmp1 tmp2 tmp3 tmp4
%%
In = Xtr;
%In = Xte;
K = 13; %class
F = size(In,2); %feature
N = size(In,1); %sample

converged = 0;
prevLoglikelihood = Inf;
round = 0;

%initial value
GMMMu = S.mu;
GMMSigma = S.Sigma;
GMMpro = S.ComPro;

while (converged ~= 1)
    round = round +1;
    disp(round);
    
    %weight
    p = zeros(N,K);
    for k = 1:K
        p(:,k) = GMMpro(k)* mvnpdf(In,GMMMu(k,:),GMMSigma(:,:,k)); 
    end
    pp = p ./ sum(p,2);

    %proportion
    nk = nansum(pp,1);
    GMMpro = nk ./N;
    
    %mean
    temp = zeros(N,F);
    for k = 1:K
        for f = 1:F
            temp(:,f) = pp(:,k) .* In(:,f);
        end
        GMMMu(k,:) = nansum(temp,1)./nk(k);
    end

    %sigma
    temp = zeros(F,F,N);
    for k = 1:K
        Xmu = (In - GMMMu(k,:));
        Xmutrans = Xmu';
        for n = 1:N
            temp(:,:,n) = pp(n,k) .* Xmutrans(:,n)*Xmu(n,:);
        end
        GMMSigma(:,:,k) = nansum(temp,3) ./ nk(k);
    end
    
    %loglikelihood
    loglikelihood = pp .* log(p);
    loglikelihood = real(loglikelihood);
    loglikelihood(~isfinite(loglikelihood))=0;
    loglikelihood  = sum(loglikelihood,2);
    loglikelihood = sum(loglikelihood);
    disp(loglikelihood);
    
    if (round >20)
        converged = 1;
    end
end
%}