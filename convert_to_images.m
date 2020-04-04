function [images] = convert_to_images(patches, batch_size_per_patch, divs, offset)
%CONVERT_TO_IMAGES converts the 3D patch data (Nb, Nx, Ny) into collection of 2D images (Nx_orig, Ny_orig, Ni). 
% Inputs:
% 	patches: 				3D patch data (Nb x Nx x Ny)
% 								Nb = Number of total patches
% 								Nx = Size of the patch in X dimension
% 								Ny = Size of the patch in Y dimension
%	batch_size_per_patch:	Number of patches used to construct a single image
%	divs:					1x2 division array used to split the image in X,Y dimension
%	offset:					1x2 array containing number of pixels addded at the begining and end of X-Y dimension
%
% Output:
%	images:					Collection of 2D images represented with a 3D (Nx_orig, Ny_orig, Nt):
%								Nx_orig = Size of the original image in X dimension
%								Ny_orig = Size of the original image in Y dimension
%								Nt      = Total number of images constructed by all patches (Nb)

[Nb, Nx, Ny] = size(patches);

images  = zeros(Nb/batch_size_per_patch, (Nx-2*offset(1))*divs(1), (Ny-2*offset(2))*divs(2));

for n = 1: size(images,1)
    interval = (n-1)*batch_size_per_patch + 1: n*batch_size_per_patch;
    
    images(n,:,:) = get_im_from_patches_2d(patches(interval,:,:), divs, offset);
    
    
end

images = permute(images, [2 3 1]);

end


function [im] = get_im_from_patches_2d(patches3d, divs, offset)
%GET_IM_FROM_PATCHED_2D reconstructs a single 2D image from its non-overlapping patches represented with 
% 3D array (Np x Nx x Ny).

sh_3d  = size(patches3d);

new_shape = (sh_3d(2:end) - offset.*2).*divs;

im = zeros(new_shape);

shape = size(im);

widths = shape./divs;

ind = 1;

for x = 1:widths(1):shape(1)
    for y = 1:widths(2):shape(2)
        patch  = squeeze(patches3d(ind,:,:,:,:));
        im(x:x+widths(1)-1,y:y+widths(2)-1) = patch(offset(1)+1:offset(1)+widths(1), ...
            offset(2)+1:offset(2)+widths(2));
        ind = ind+1;
        
    end
end

end


