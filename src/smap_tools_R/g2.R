g2 <- function(XYZ, beta = c(1, 0.5)) {
  A <- beta[1]
  sigma <- beta[2]
  A * exp(-(XYZ^2) / (2 * sigma^2))
}
