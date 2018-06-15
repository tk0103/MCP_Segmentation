%Parameter
crbin = 0.5; rbin = 0.25; radii = 15:rbin:34; numofcircle = 12; range = 55; numofrad = size(radii,2);
sta = 1; send = 50; rlimit = 1.0; crlimit = 1.5;

%ini
EdgeImg = zeros(siz); [xx,yy] = ndgrid(1:(siz(1)/crbin),1:(siz(2)/crbin));
theta = linspace(0,2*pi,360); peaksall = zeros(numofcircle,3,siz(3));

%Canny edge detection  
for snum = sta:send
    SliceImg = Img(:,:,snum);
    SliceImg(SliceImg<0.14) = 0;
    EdgeImg(:,:,snum) = edge(SliceImg, 'canny',[],1.7);
end
%%
%Display
imagesc(EdgeImg(:,:,10)');
axis tight equal off
colormap gray

%%
%Save
save_raw(EdgeImg,[InputPath CasePath{1,:} '_Output' '.raw'],'*single');
