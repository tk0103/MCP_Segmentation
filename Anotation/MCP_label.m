%Parameter
crbin = 0.5; rbin = 0.25; radii = 15:rbin:34; numofcircle = 12; range = 55; numofrad = size(radii,2);
sta = 1; send = 50; rlimit = 1.0; crlimit = 1.5;

%ini
EdgeImg = zeros(siz); [xx,yy] = ndgrid(1:(siz(1)/crbin),1:(siz(2)/crbin));
theta = linspace(0,2*pi,360); peaksall = zeros(numofcircle,3,siz(3));

for snum = sta:send
    disp(snum);
    d2Edge = EdgeImg(:,:,snum);
    index = find(d2Edge==1);
    HoughImg = zeros(siz(1)/crbin,siz(2)/crbin,numofrad);
    peak = zeros(numofcircle,3);
    
    %Create HoughImg
    for indvalue = 1:size(index)
        [x,y] = ind2sub([siz(1) siz(2)],index(indvalue));
        cx = x - radii' * cos(theta); cy = y + radii' * sin(theta);
        cx = round(cx/crbin); cy = round(cy/crbin);
        for rnum = 1:numofrad
            for bb = 1:360
                HoughImg(cx(rnum,bb),cy(rnum,bb),rnum) = HoughImg(cx(rnum,bb),cy(rnum,bb),rnum) + 1;
            end
        end      
    end
    
    %
    if snum == sta
        for cnum = 1:numofcircle
            mask = false(size(HoughImg));
            [M,I] = max(HoughImg(:));
            [cx,cy,cr] = ind2sub(size(HoughImg),I);
            peak(cnum,:) = [cx cy radii(cr)];
            
            temp =(xx-cx).^2 + (yy-cy).^2 <= (range/crbin).^2;
            mask(:,:,1:numofrad) = repmat(temp,[1 1 numofrad]);
            HoughImg(mask) = 0;
        end
        peak = sortrows(peak,1);
        peaksall(:,:,snum) = peak;
        
    else
        for cnum = 1:numofcircle
                region = zeros(size(HoughImg));
                mask = false(size(HoughImg));
                minnum = (peaksall(cnum,3,snum-1)-radii(1))/rbin +1 - rlimit/rbin;
                maxnum = (peaksall(cnum,3,snum-1)-radii(1))/rbin +1 + rlimit/rbin;
                if maxnum > size(radii,2) 
                    maxnum = size(radii,2); end
                if minnum < 1
                    minnum = 1; end
                
                temp =(xx-peaksall(cnum,1,snum-1)).^2 + (yy-peaksall(cnum,2,snum-1)).^2 <= (crlimit/crbin).^2;
                mask(:,:,minnum:maxnum) = repmat(temp,[1 1 maxnum-minnum+1]);

                region(mask) = HoughImg(mask);
                [M,I] = max(region(:));
                [cx,cy,cr] = ind2sub(size(HoughImg),I);
                peak(cnum,:) = [cx cy radii(cr)]; 
        end
        peaksall(:,:,snum) = peak;
    end
end
peaksall(:,1:2,:) = peaksall(:,1:2,:)*crbin;

%%
%figure
snum =30;
imagesc(Img(:,:,snum)');
axis tight equal off
colormap gray
viscircles(peaksall(:,1:2,snum), peaksall(:,3,snum),'EnhanceVisibility',false,'EdgeColor','r','LineWidth',1);
%viscircles(robustresult(:,1:2,snum), robustresult(:,3,snum),'EnhanceVisibility',false,'EdgeColor','r','LineWidth',1);

%%
%Robutst linear regression
slice = sta:1:send; slice = slice(:);
robustresult = zeros(numofcircle,3,siz(3));

for cnum =1:numofcircle
    for bb = 1:3
        vt = peaksall(cnum,bb,sta:send); vt = vt(:);
        tbl = table(slice,vt,'VariableNames',{'x','y'});
        lm = fitlm(tbl,'y~x','RobustOpts','on');
        robustresult(cnum,bb,sta:send) = lm.Fitted;
    end
end

%No1,2,radius,median
robustresult(1,3,sta:send) = median(peaksall(1,3,sta:send));
robustresult(2,3,sta:send) = median(peaksall(2,3,sta:send));


%%
%ƒ‰ƒxƒ‹‰æ‘œì¬
[xx,yy] = ndgrid(1:siz(1),1:siz(2));
groundtruth = zeros(siz);
%houghval = peaksall;
houghval = robustresult;

for snum = sta:send
    temp = zeros(siz(1),siz(2));
    for cnum = 1:numofcircle
        mask = (xx-houghval(cnum,1,snum)).^2 + (yy-houghval(cnum,2,snum)).^2 <= houghval(cnum,3,snum)^2;
        temp(mask) = cnum;
    end
    groundtruth(:,:,snum) = temp;
end

%groundtruth = permute(groundtruth,[2 1 3]);
save_raw(groundtruth,'C:\Users\yourb\Desktop\2groundtruth0.50.25robust.raw','*single');

%%
%‰~‰æ‘œ
hougheval = robustresult;
CircleImg = zeros(siz);
for snum = sta:send
    for cnum = 1:numofcircle
        r = hougheval(cnum,3,snum);
        cx = hougheval(cnum,1,snum);
        cy = hougheval(cnum,2,snum);
        x = r*sin(theta)+cx;
        y = r*cos(theta)+cy;
        x = round(x); y = round(y);
        for bb = 1:360
            CircleImg(y(bb),x(bb),snum) = 1;
        end

    end
end
CircleImg = permute(CircleImg,[2 1 3]);
save_raw(CircleImg,'C:\Users\yourb\Desktop\CircleImg05025.raw','*single');
%%
%•]‰¿
houghval = peaksall;
%houghval = robustresult;
[xx,yy] = meshgrid(1:siz(1),1:siz(2));
DistImg = zeros(siz);
Dtem = ones(siz(1),siz(2))*10000; Dtem = Dtem(:);

for snum = sta:send
    for cnum = 1:numofcircle
        Ddash = abs(sqrt((xx-houghval(cnum,1,snum)).^2 + (yy-houghval(cnum,2,snum)).^2) - houghval(cnum,3,snum));
        Ddash = Ddash(:);
        Dtem(Dtem > Ddash) = Ddash(Dtem > Ddash);
    end
    DistImg(:,:,snum) = reshape(Dtem,[siz(1) siz(2)]);
 end

%highlight image
canny_eval = load_raw('C:\Users\yourb\Desktop\Canny_evaluation2.raw','*single');
canny_eval = reshape(canny_eval,siz); canny_eval = logical(canny_eval);

snum =15;
Dtemp = DistImg(:,:,snum)';
disp(mean(Dtemp(canny_eval(:,:,snum))));

%%
[p,h,stats] = signrank(c,b);
boxplot([a,b,c,d,e],'Labels',{'cb=0.25,rb=0.25','cb=0.5,rb=0.25','cb=0.25,rb=0.5','cb=0.5,rb=0.5','cb=1,rb=1'});
%%
%figure
matnum = 9;

hold on
yyaxis left
plt1 = robustresult(matnum,1,sta:send);
plot(slice,plt1(:),'--');
scatter(slice',peaksall(matnum,1,sta:send),'+');
xlabel('slice number'); ylabel('Coordinate x');

plt2 = robustresult(matnum,2,sta:send);
yyaxis right
plot(slice,plt2(:),'-.');
scatter(slice',peaksall(matnum,2,sta:send),'x');
ylabel('Coordinate y');
hold off
figure;

plt3 = robustresult(matnum,3,sta:send);
plot(slice,plt3(:),'--');
xlabel('slice number'); ylabel('Radius');
hold on
scatter(slice',peaksall(matnum,3,sta:send),'+');
hold off


