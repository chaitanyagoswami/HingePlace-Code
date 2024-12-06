clc;
clear;
%% Loading Brain Matrices
load("example/MNI152_T1_1mm_MNI152CustomleadField_roastResult.mat",'A_all')
load("example/MNI152_T1_1mm_MNI152CustomleadField.mat")
load("example/MNI152_T1_1mm_ras_header.mat")
%% Indices for Brain Elements
indBrain = elem((elem(:,5)==1 | elem(:,5)==2),1:4); indBrain = unique(indBrain(:));
A_brain = A_all(indBrain,:,:);
nodeV = zeros(size(node,1),3);
for i=1:3, nodeV(:,i) = node(:,i)/hdrInfo.pixdim(i); end
locs_brain = nodeV(indBrain,1:3);
save('forward_matrices/Aall.mat', "A_brain");
save('forward_matrices/loc_all.mat', "locs_brain");