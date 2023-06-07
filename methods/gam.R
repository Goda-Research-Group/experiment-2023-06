set.seed(1234)

library(mgcv)

args <- commandArgs(trailingOnly = T)
tmpdir = args[1]

x <- read.table(paste(tmpdir, "/inputX.txt", sep = ""))
y <- read.table(paste(tmpdir, "/inputY.txt", sep = ""))

xSize <- ncol(x)
ySize <- ncol(y)
N <- nrow(x)

g.mat <- matrix(NA, N, xSize)

for (i in 1:xSize){
    if (ySize == 1) {
        x.gam <- gam(x[, i] ~ te(y[, 1]))
    } else if (ySize == 2) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]))
    } else if (ySize == 3) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]))
    } else if (ySize == 4) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]))
    } else if (ySize == 5) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]))
    } else if (ySize == 6) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]) + te(y[, 6]))
    } else if (ySize == 7) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]) + te(y[, 6]) + te(y[, 7]))
    } else if (ySize == 8) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]) + te(y[, 6]) + te(y[, 7]) + te(y[, 8]))
    } else if (ySize == 9) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]) + te(y[, 6]) + te(y[, 7]) + te(y[, 8]) + te(y[, 9]))
    } else if (ySize == 10) {
        x.gam <- gam(x[, i] ~ te(y[, 1]) + te(y[, 2]) + te(y[, 3]) + te(y[, 4]) + te(y[, 5]) + te(y[, 6]) + te(y[, 7]) + te(y[, 8]) + te(y[, 9]) + te(y[, 10]))
    }
    g.mat[, i] <- x.gam$fitted
}

g <- as.data.frame(g.mat)

write.table(g, paste(tmpdir, "/outputX.txt", sep = ""), sep = " ", row.names = F, col.names = F)
