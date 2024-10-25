function p3a(varargin);

figure(gcf); hold on;
or  = zeros(1,3);
ax  = eye(3);
alx = zeros( 1, 3, 3 );
ax=ax.*100;
plot3( [ or; ax(1,:) ], [ or ; ax(2,:) ], [ or; ax(3,:) ],'LineWidth',2 ); hold on;
axis equal; grid on;
