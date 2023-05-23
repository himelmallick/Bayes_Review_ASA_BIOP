##########################################
### Use the new PNAS paper data as an example for 
### Bayesian nonclinical working group
### By Yushi Liu and Zhe Sun
############################################
setwd("/lrlhps/users/c184067/Bayesian_non_clinical_group/data_for_himel/")
exp.dat<-read.csv("MicroarrayDataBrainProstateStandardized9895.csv",header=T)
colnames(exp.dat)[1]<-"probe"
label.tumor<-t(exp.dat[1,-1])
colnames(label.tumor)<-"tumor.type"
label.pten<-t(exp.dat[2,-1])
colnames(label.pten)<-"pTenstat"
gene.summary<-read.csv("GeneSummary.csv",header=T)
gene.exp<-exp.dat[-c(1:2),]

n.noise=5
gene.summary.feature<- gene.summary %>% arrange(desc(p.ttest)) %>% dplyr::filter(p.ttest<0.001) %>% mutate(signals=1)
gene.summary.noise<-gene.summary %>% arrange(desc(p.ttest)) %>% dplyr::filter(p.ttest>0.8) %>% mutate(signals=0)
gene.summary.noise.rand<-gene.summary.noise[sample.int(length(gene.summary.noise$p.ttest),n.noise),]
final.candiate<-rbind(gene.summary.feature,gene.summary.noise.rand) 
temp.exp.dat<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(-probe)
temp.exp.probes<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(probe)
final.exp.dat <-t(scale(t(temp.exp.dat)))
row.names(final.exp.dat)<-temp.exp.probes$probe
dat.pca <- prcomp(t(final.exp.dat))
pc1.weight<--1*dat.pca$rotation[,1,drop=F]
pc1<-t(final.exp.dat) %*% pc1.weight
#plot(density(pc1),main="Density plot of PC1 using all the genes")
library("bclust")
library("diptest")
bcdx.bclust<-bclust(t(final.exp.dat),var.select = T,transformed.par=c(0,-50,0 ,0,0,0))

#bcdx.bclust<-bclust(t(final.exp.dat),var.select = T,transformed.par=c(0,-50,0 ,1,0,0),labels=label.pten)
plot(as.dendrogram(bcdx.bclust))
k.res<-kmeans(t(final.exp.dat),2)$cluster
k.res.df<-data.frame(cluster=k.res,sample=names(k.res))
res<-data.frame(cluster=cutree(bcdx.bclust,2),sample=names(cutree(bcdx.bclust,2)))
label.stat<-data.frame(status=label.pten,sample=row.names(label.pten))
k.label.stat<-merge(k.res.df,label.stat)
all.res<-merge(res,label.stat)
table(all.res$pTenstat,all.res$cluster)
table(k.label.stat$pTenstat,k.label.stat$cluster)
plot(bcdx.bclust$clust.number,bcdx.bclust$logposterior,
     xlab="Number of clusters",ylab="Log posterior",type="b")
abline(h=max(bcdx.bclust$logposterior))

bcdx.imp<-imp(bcdx.bclust)
bcdx.imp.df<-data.frame(var=bcdx.imp$var,ProbeSet=bcdx.imp$labels)
bcdx.imp.df<-merge(bcdx.imp.df,final.candiate)
write.csv(bcdx.imp.df,"/lrlhps/users/c184067/Bayesian_non_clinical_group/data_for_himel/tabl1.csv")

dip.res<-apply(gene.exp[,-1],1,function(x) dip.test(x)$p.value)
dip.df<-data.frame(ProbeSet=gene.exp$probe,dip.p=dip.res)
all.dip.gene.sum<-merge(gene.summary,dip.df,by="ProbeSet")


#############################################################
##  Sensitivity analysis to include different numbers of noise genes
##############################################################
library("bclust")
setwd("/lrlhps/users/c184067/Bayesian_non_clinical_group/data_for_himel/")
exp.dat<-read.csv("MicroarrayDataBrainProstateStandardized9895.csv",header=T)
colnames(exp.dat)[1]<-"probe"
label.tumor<-t(exp.dat[1,-1])
colnames(label.tumor)<-"tumor.type"
label.pten<-t(exp.dat[2,-1])
colnames(label.pten)<-"pTenstat"
gene.summary<-read.csv("GeneSummary.csv",header=T)
gene.exp<-exp.dat[-c(1:2),]


gene.summary.feature<- gene.summary %>% arrange(desc(p.ttest)) %>% dplyr::filter(p.ttest<0.001) %>% mutate(signals=1)
gene.summary.noise<-gene.summary %>% arrange(desc(p.ttest)) %>% dplyr::filter(p.ttest>0.8) %>% mutate(signals=0)
final.res<-data.frame(matrix(ncol=2,nrow=0))
for (n.noise in c(rep(seq(5,100,5),100))){
gene.summary.noise.rand<-gene.summary.noise[sample.int(length(gene.summary.noise$p.ttest),n.noise),]
final.candiate<-rbind(gene.summary.feature,gene.summary.noise.rand) 
temp.exp.dat<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(-probe)
temp.exp.probes<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(probe)
final.exp.dat <-t(scale(t(temp.exp.dat)))
row.names(final.exp.dat)<-temp.exp.probes$probe

bcdx.bclust<-bclust(t(final.exp.dat),var.select = T,transformed.par=c(0,-50,0 ,0,0,0))
res<-data.frame(cluster=cutree(bcdx.bclust,2),sample=names(cutree(bcdx.bclust,2)))
label.stat<-data.frame(status=label.pten,sample=row.names(label.pten))
all.res<-merge(res,label.stat)
temp.df<-as.data.frame(table(all.res$pTenstat,all.res$cluster))
perform1<-(temp.df[1,"Freq"]+temp.df[4,"Freq"])/25
perform2<-(temp.df[2,"Freq"]+temp.df[3,"Freq"])/25
final.res<-rbind(final.res, data.frame(n.noise=n.noise, perform=max(perform1,perform2)))
}
tapply(final.res[,"perform"],final.res[,"n.noise"],mean)

library(gridExtra)
library(ggplot2)
g1 <- ggplot(final.res, aes(factor(n.noise), perform)) + geom_boxplot() +
  ggtitle("Bayesian Clustering Performance vs. Number of Noise Genes") + xlab("Number of Noise Genes") +ylab("Performance (Accuracy)")
plot(g1)


k.final.res<-data.frame(matrix(ncol=2,nrow=0))
for (n.noise in c(rep(seq(5,100,5),100))){
  gene.summary.noise.rand<-gene.summary.noise[sample.int(length(gene.summary.noise$p.ttest),n.noise),]
  final.candiate<-rbind(gene.summary.feature,gene.summary.noise.rand) 
  temp.exp.dat<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(-probe)
  temp.exp.probes<-gene.exp %>% dplyr::filter(probe %in% final.candiate$ProbeSet) %>%dplyr::select(probe)
  final.exp.dat <-t(scale(t(temp.exp.dat)))
  row.names(final.exp.dat)<-temp.exp.probes$probe
  
  k.res<-kmeans(t(final.exp.dat),2)$cluster
  k.res.df<-data.frame(cluster=k.res,sample=names(k.res))
  label.stat<-data.frame(status=label.pten,sample=row.names(label.pten))
  k.res<-merge(k.res.df,label.stat)
  temp.df<-as.data.frame(table(k.res$pTenstat,k.res$cluster))
  perform1<-(temp.df[1,"Freq"]+temp.df[4,"Freq"])/25
  perform2<-(temp.df[2,"Freq"]+temp.df[3,"Freq"])/25
  k.final.res<-rbind(k.final.res, data.frame(n.noise=n.noise, perform=max(perform1,perform2)))
}
tapply(k.final.res[,"perform"],k.final.res[,"n.noise"],mean)
