%% Saving the plots as images

close all;
clear all;
files = dir('*.mat'); %collect all the .mat files

for k=2:1001 %we start at 2 so we do not consider the mat1616 file
    filename = files(k).name;
    load(filename);
    
    %normalizing the data
    pos(:,1)=(pos(:,1)-min(pos(:,1)))/(max(pos(:,1))-min(pos(:,1)));
    pos(:,2)=(pos(:,2)-min(pos(:,2)))/(max(pos(:,2))-min(pos(:,2)));
    
    %we keep only the first two dimensions
    plot(pos(:,1),pos(:,2));
    
    %we enlarge a bit the graph so the plot does not match with the axes
    %in order to remove them later
    axis([-0.1 1.1 -0.1 1.1]);
    
    %we save the graph as an image
    F = getframe;
    A = F.cdata(:,:,1);
    [n,m]=size(A);
    
    % Obtaining an image with only two colors (black 1 and white 255)
    for i = 1:n
        for j = 1:m
            a = A(i,j);
            if a<200
                A(i,j)=1;
            else
                A(i,j)=255;
            end
        end
    end
    
    % Removing the border of the graph
    for i = 1:n
        for j= 1:m
            if i>10 & j>10 & (n-i)>10 & (m-j)>10
                B(i-10,j-10)=A(i,j);
            end
        end
    end
    
    filename = strcat(erase(filename,'mat'),'jpg'); %Restructure the file name
    imwrite(B,filename); %save the image in the current folder
end

%% Saving the data in a 2-dimensionnal matrix

close all;
clear  all;
files = dir('*.jpg'); %collect all .jpg images that we just saved

mat1616 = zeros(256,1000); %each line of mat1616 corresponds to one image
% of a sample that has been resize in a 16x16 image and stock in a single
% vector

for i = 1:1000
    img =  imread(files(i).name); %we read the image
    res = imresize(img,[16 16]); %we resize it into an 16x16 image
    mat1616(:,i) = double(res(:)); %we store it in mat1616 as one vector of length 256
end

save('mat1616.mat','mat1616'); %we save the matrix to use it for classification