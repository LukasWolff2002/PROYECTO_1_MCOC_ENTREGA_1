Lx = 1;
Ly = 1;
Nx = 100;
Ny = 100;
nx = Nx+1;
ny = Ny +1;
dx = Lx/Nx;
dy = Ly/Ny

x=(0:Nx)*dx
y=(0:Ny)*dy


boundary_index = [1:nx, 1:nx:(ny-1)*nx, 1+(ny-1)*nx:nx*ny, nx:nx:nx*ny]


diagonals = [4*ones(nx*ny,1), -ones(nx*ny,4)];
A = spdiags(diagonals, [0 -1 1 -nx nx], nx*ny, nx*ny);
I = speye(nx*ny);

A(boundary_index, :)= I(boundary_index, :);

b = zeros(nx,ny);
b(:,1) = 0;
b(1,:)= 0;
b(:,ny) = 4*x.*(1-x);
b(nx,:)=0;
b=reshape(b,nx*ny,1);

%Se resuelve la ecuacion de laplace usando la eliminacion gausiana
Phi = A\b;
Phi = reshape(Phi,nx,ny);

%Finalmente vamos a los graficos
[X,Y]=meshgrid(x,y);
v=[0.8 0.6 0.4 0.2 0.1 0.05 0.01];
contour(X,Y,Phi,v, 'ShowText', 'on');
axis equal;
set(gca, 'Ytick', [0 0.2 0.4 0.6 0.8 1]);
set(gca, 'Xtick', [0 0.2 0.4 0.6 0.8 1]);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 14);
title('Titulo', 'Interpreter', 'latex', 'FontSize', 14);