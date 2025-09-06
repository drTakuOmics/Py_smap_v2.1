# Rotate 2-D and 3-D arrays using simple nearest-neighbour interpolation

rotate2dMatrix <- function(img, R) {
  if (all(dim(R) == c(3,3))) {
    R <- R[1:2,1:2]
  }
  n <- dim(img)
  center <- (n - 1) / 2
  grid <- expand.grid(x = 0:(n[1]-1), y = 0:(n[2]-1))
  coords <- as.matrix(grid)
  coords_centered <- sweep(coords, 2, center, FUN = "-")
  src <- coords_centered %*% t(R) + matrix(center, nrow(coords_centered), 2, byrow = TRUE)
  src <- round(src)
  valid <- src[,1] >= 0 & src[,1] < n[1] & src[,2] >= 0 & src[,2] < n[2]
  out <- array(0, dim = n)
  out[cbind(grid$x[valid] + 1, grid$y[valid] + 1)] <- img[cbind(src[valid,1] + 1, src[valid,2] + 1)]
  out
}

rotate3dMatrix <- function(vol, R) {
  n <- dim(vol)
  center <- (n - 1) / 2
  grid <- expand.grid(x = 0:(n[1]-1), y = 0:(n[2]-1), z = 0:(n[3]-1))
  coords <- as.matrix(grid)
  coords_centered <- sweep(coords, 2, center, FUN = "-")
  src <- coords_centered %*% t(R) + matrix(center, nrow(coords_centered), 3, byrow = TRUE)
  src <- round(src)
  valid <- src[,1] >= 0 & src[,1] < n[1] & src[,2] >= 0 & src[,2] < n[2] & src[,3] >= 0 & src[,3] < n[3]
  out <- array(0, dim = n)
  out[cbind(grid$x[valid] + 1, grid$y[valid] + 1, grid$z[valid] + 1)] <- vol[cbind(src[valid,1] + 1, src[valid,2] + 1, src[valid,3] + 1)]
  out
}
