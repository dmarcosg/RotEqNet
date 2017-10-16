function fr = rotateVectorField(f,alpha)
% fr = rotateVectorField(f,alpha)
% like imrotate for vector fields
% alpha in degrees

%for m = 1:size(f,5)
f= imrotate(f,alpha,'bilinear','crop');
%end

fr = f;
if size(f,5) == 2
    cos_a = cos(alpha*pi/180);
    sin_a = sin(alpha*pi/180);
    fr(:,:,:,:,1) = f(:,:,:,:,1)*cos_a - f(:,:,:,:,2)*sin_a;
    fr(:,:,:,:,2) = f(:,:,:,:,2)*cos_a + f(:,:,:,:,1)*sin_a;
end