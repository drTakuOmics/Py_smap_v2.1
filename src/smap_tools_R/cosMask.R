# Radial distance helper equivalent to MATLAB rrj
rrj <- function(mat) {
  dims <- dim(mat)
  stopifnot(length(dims) %in% c(2,3))
  center <- (dims + 1) / 2
  coords <- lapply(dims, function(d) seq_len(d) - center[which(dims==d)[1]])
  grid <- as.matrix(expand.grid(coords))
  rr <- rowSums(grid^2)
  array(sqrt(rr), dim = dims)
}

# Create a variable cosine mask similar to MATLAB cosMask
variableCosMask <- function(edge, width) {
  r <- rrj(array(0, dim = c(edge, edge)))
  mask <- r <= edge/2
  transition <- r > edge/2 & r < edge/2 + width
  mask[transition] <- 0.5 * (1 + cos(pi * (r[transition] - edge/2) / width))
  mask[r >= edge/2 + width] <- 0
  mask
}
