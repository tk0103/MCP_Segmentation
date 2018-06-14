%Input
img = load_raw('C:\Users\yourb\Desktop\NZdata\MCP\3Dvolume\MCP_phantom12E4.raw','*single');
siz = [544,544,50]; img = reshape(img,siz);
ringartifact = 0;
%%
figure;
imagesc(img(:,:,50)');
axis tight equal 
colormap default 
%caxis([0 4000])
%%
imagesc(img(:,:,30)');
axis tight equal off
colormap gray

%%
imagesc(HA(:,:,41)');
axis tight equal
colormap gray
%%
if (ringartifact == 0)
    img_cpu = img;  ldash = zeros(siz);
    alpha  =0;   beta = 0;  gamma = 0;
    sig_d = 1.0;    sig_r = 0.6;    sig_h = 0.1;
    
else
    alpha  = 8000;  beta = 100000;  gamma = 4;
    sig_d = 1.0;    sig_r = 0.6;    sig_h = 0.005;
    %center = 240.5; %Center coordinate
    center = 295;
    s = 1.5; %sigma of gaussian
    w = 6 * s;
    
    [grx,gry,grz] =ndgrid(-w:w,-w:w,-w:w);
    g = 1 / ((2* pi)^(3/2)*s^7) * exp(- (grx.^2 + gry.^2 + grz.^2) / (2*s^2)) ;

    %Gaussian
    gxx = g .* (grx.^2 - s^2);
    gxy = g.* (grx.* gry);
    gxz = g.*(grx.* grz);
    gyy = g.* (gry.^2 - s^2);
    gyz = g.*(gry.*grz);
    gzz = g.*(grz.^2 - s^2);
    
    img_cpu = img;
    img = gpuArray(img);
    ixx = gather(imfilter(img,gxx,'replicate'));
    ixy = gather(imfilter(img,gxy,'replicate'));
    ixz = gather(imfilter(img,gxz,'replicate'));
    iyy = gather(imfilter(img,gyy,'replicate'));
    iyz = gather(imfilter(img,gyz,'replicate'));
    izz = gather(imfilter(img,gzz,'replicate'));

    %Angle and vector
    [x,y,z] = ndgrid((1:siz(1))-center,(1:siz(2))-center,1:siz(3));
    theta = atan2(y(:),x(:));
    v3 = [-sin(theta),cos(theta),zeros(size(theta))];
    v2 = [cos(theta),sin(theta),zeros(size(theta))];
    v1 = [zeros(size(theta)),zeros(size(theta)),ones(size(theta))];
    
    %Definition of É…
    I1 = [ixx(:),ixy(:),ixz(:)];
    I2 = [ixy(:),iyy(:),iyz(:)];
    I3 = [ixz(:),iyz(:),izz(:)];
    F = @(v) abs(dot( [dot(I1,v,2),dot(I2,v,2),dot(I3,v,2)],v,2));
    lambda3 = F(v3);
    lambda2 = F(v2);
    lambda1 = F(v1);
    l3 = reshape(lambda3,siz);
    l2 = reshape(lambda2,siz);
    l1 = reshape(lambda1,siz);
    
    ldash = (l1 - l3);
    ldash(ldash < 0) = 0;
    %ldash = ldash.^2;
    
    %dilation
    
    nhood = zeros(3,3,3);
    nhood(2,2,2) = 1.0;
    for a = -1:1
        for b = -1:1
            for center= -1:1
                if(1<=abs(a)+abs(b)+abs(center) && abs(a)+abs(b)+abs(center)<=2)
                    nhood(a+2,b+2,center+2) = 1.0;
                end
            end
        end
    end
    SE = strel('arbitrary',nhood);
    ldash = imdilate(ldash,SE);
    
end
%%
    alpha  = 0;  beta = 0;  gamma = 4;
    sig_d = 1.0;    sig_r = 5000;    sig_h = 0.1;
    %%
%figure;
imagesc(ldash(:,:,50));
axis equal tight
%caxis ([0,4000])
%%
%pre calculate
In = img_cpu;
w = 9.0;
[X,Y,Z] = meshgrid(-w:w,-w:w,-w:w);
xyz = (X.^2+Y.^2+Z.^2);
dim = size(In);
Out = zeros(dim);
Outdash = zeros(dim);
Thir = exp(-ldash.^2 / (2*sig_h^2));
powerldash = ldash.^2;

%èàóùäJén
for k = 28:33
    disp(k);
    for j = 1:siz(2) %x
        for i = 1:siz(1) %y
             
         % Extract local region
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         kMin = max(k-w,1);
         kMax = min(k+w,dim(3));
         
         I = In(iMin:iMax,jMin:jMax,kMin:kMax);
         Thir_tmp = Thir(iMin:iMax,jMin:jMax,kMin:kMax);
         sig_ddash = powerldash(i,j,k).*alpha + sig_d;
         sig_rdash = powerldash(i,j,k).*beta + sig_r;
         
         % Compute Gaussian intensity weights
         Fir = exp(-xyz((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1,(kMin:kMax)-k+w+1)./(2*sig_ddash.^2));
         Sec = exp(-(I-In(i,j,k)).^2./(2*sig_rdash.^2));
         
         % Calculate bilateral filter response
         FST = Fir.*Sec.*(Thir_tmp).^gamma;
         Out(i,j,k) = sum(FST(:).*I(:))/sum(FST(:));
    
        end 
    end
end
%%
%save_raw(Out,'C:\Users\yourb\Desktop\PBL_MCP_phantom12E4.raw','*single');
figure;
imagesc(Out(:,:,30)');
axis tight equal off
colormap gray
