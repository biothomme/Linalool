setwd("~/Desktop")
scedat <- read.table("posterscent.csv", sep=";", header=TRUE, dec=",", colClasses = "numeric")
install.packages("vegan")
library(MASS)
library(vegan)
# # try <- as.matrix(t(scedat))
# # rownames(scedat) <- try[1,]
# # tryy <- subset(try, try[,1]!=try[1,1])
# # tryy <- as.data.frame(tryy)

sce.dat <- sqrt(sqrt(scedat))

# pladis <- vegdist(sce.dat)
# 
# colMeans(scedat[,])
# 
# sce.mds0 <- isoMDS(pladis)
# stressplot(sce.mds0, pladis)
# ordiplot(sce.mds0, type = "t")

sce.mds <- metaMDS(sce.dat, trace = F)

plot(sce.mds, type = "t", cex = c(1.6*colMeans(sce.dat)), xlim=c(-0.3,0.6), ylim=c(-0.3,0.3))
"points"(sce.mds, display = c("sites", "species"), choices = c(1, 2), pch =16, cex=3, shrink = FALSE, col= c(4,4,4,4,4,3,3,3))
fval.scent <- adonis2(sce.dat ~ c(1,1,1,1,1,2,2,2), permutations = 99999, method = "bray")
# sce.mds0 <- cmdscale(pladis)
# ordiplot(sce.mds0, type="t")

# scores(sce.mds, display = "species")

