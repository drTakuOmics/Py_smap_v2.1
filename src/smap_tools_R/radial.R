radialMean <- function(im) {
  im <- as.matrix(im)
  nr <- nrow(im); nc <- ncol(im)
  cx <- (nc - 1) / 2; cy <- (nr - 1) / 2
  x <- matrix(rep(0:(nc - 1), each = nr), nr, nc)
  y <- matrix(rep(0:(nr - 1), nc), nr, nc)
  r <- round(sqrt((x - cx)^2 + (y - cy)^2))
  vals <- tapply(im, r, mean)
  maxr <- max(r)
  out <- numeric(maxr + 1)
  out[as.integer(names(vals)) + 1] <- vals
  out
}

radialAverage <- function(im) {
  prof <- radialMean(im)
  nr <- nrow(im); nc <- ncol(im)
  cx <- (nc - 1) / 2; cy <- (nr - 1) / 2
  x <- matrix(rep(0:(nc - 1), each = nr), nr, nc)
  y <- matrix(rep(0:(nr - 1), nc), nr, nc)
  r <- round(sqrt((x - cx)^2 + (y - cy)^2))
  matrix(prof[r + 1], nr, nc)
}

radialMax <- function(im) {
  im <- as.matrix(im)
  nr <- nrow(im); nc <- ncol(im)
  cx <- (nc - 1) / 2; cy <- (nr - 1) / 2
  x <- matrix(rep(0:(nc - 1), each = nr), nr, nc)
  y <- matrix(rep(0:(nr - 1), nc), nr, nc)
  r <- round(sqrt((x - cx)^2 + (y - cy)^2))
  vals <- tapply(im, r, max)
  maxr <- max(r)
  out <- rep(-Inf, maxr + 1)
  out[as.integer(names(vals)) + 1] <- vals
  out
}
