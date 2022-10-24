# This script is able to compute PCA for the primula shape dataset and output csv 
# as useful for the plotting.
# author: thomas huber
# mail: thoms.huber@evobio.eu

# packages
devtools::install_github("MomX/Momocs")

install.packages("rlang")
library(Momocs)
library(ggplot2)

# constants
PRIMI = c(rep('hirsuta', 14),rep('lutea', 10), rep('hirsuta', 12))

# main
setwd('/Users/Thomsn/Desktop/posterscent/morphom/rap')
files = list.files()
files = files[c(2:length(files))]
shape_s = list()
lmrk_s = list()
for (file in files){
  shape_s[length(shape_s)+1] = import_txt(file)
  shap_e = read.table(file)
  centers = apply(shap_e, 1, function(x) sqrt(x[1]**2 + x[2]**2))
  centern = sort(centers)
  for (ce in centern[1]){
    wh = which(centers == ce)
    ces = c(wh -10, wh, wh+10)
  }
  lmrk_s[[length(lmrk_s)+1]] = ces
}

out_s = Out(shape_s, fac = dplyr::data_frame(PRIMI), ldk = lmrk_s)

# save first two axes
coe_s = out_s %>%                  
  fgProcrustes() %>%           
  coo_slide(ldk = 2) %>%                  
  efourier(30) %>%
  PCA()
pca_data = as.data.frame(coe_s$x[,1:2])
pca_data$species = as.character(coe_s$fac$primi)
write.csv(pca_data, file='pca_data.csv', sep=',')

# save percentage of explanation per PCA axis
pca_power = as.data.frame(coe_s$eig)
write.csv(pca_power, file='pca_power.csv', sep=',')

# plot the PCAs
coe_s = out_s %T>%
  stack() %>%                  
  fgProcrustes() %>%           
  coo_slide(ldk = 2) %T>%      
  stack() %>%                  
  efourier(30) %>%
  PCA() %T>%                   
  plot_PCA(~primi, axes=c(1,2)) %>% 
  LDA(~primi) %>% 
  plot_LDA(ldas)


# Plot many PCAs
bot.p <- PCA(coe_s)
PCcontrib(bot.p, nax=1:6)
gg <- PCcontrib(bot.p, nax=1:2, sd.r=c(-0.4, -0.75, -0.5, -0.25, 0, 0.5, 1, 1.5, 2.0))
gg$gg + geom_polygon(fill="slategrey", col="black") + ggtitle("A nice title")
# save new 
write.csv(gg$gg$data, file='pca_axes.csv', sep=',')
gg$gg$data$shp
