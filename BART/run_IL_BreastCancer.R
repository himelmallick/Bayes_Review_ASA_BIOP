###########################################
# Remove all variables from the workspace #
###########################################

rm(list = ls())

##################
# Load Libraries #
##################

library(foreach)
library(doParallel)
library(tidyverse)
library(SuperLearner)
library(IntegratedLearner)

###########################################
# Create an output folder if non-existent #
###########################################

if (!file.exists("Output")) {
  print("Creating output folder")
  dir.create('Output')
}

#################
# Load all data #
#################

#################
# Dataset names #
#################

outcome <- c('Basal','Her2','LumA','LumB')

for(k in 1:length(outcome)){
  
  datasets<-paste('breast',outcome[k],'train',sep='_')
  
  ###################################
  # Load Independent Validation Set #
  ###################################
  
  test_data <- paste('breast',outcome[k],'test',sep='_')
  load(paste('./Input/',test_data,'.RData', sep = ''))
  feature_table_valid<-pcl$feature_table
  sample_metadata_valid<-pcl$sample_metadata
  rm(pcl)
  
  
  ############################
  # Speciy all base learners #
  ############################
  
  base_learners<-c('SL.xgboost', 
                   'SL.randomForest', 
                   'SL.BART')   
  
  ################################
  # Speciy high-level parameters #
  ################################
  
  folds<-5
  
  ###########################
  # Loop over each modality #
  ###########################
  
  for (i in 1:length(datasets)){
    
    #################
    # Print message #
    #################
    
    cat('Running dataset:', datasets[i], '\n')
    
    ################
    # Load Dataset #
    ################
    
    load(paste('./Input/', datasets[i], '.RData', sep = ''))
    
    #################################
    # Extract individual components #
    #################################
    
    feature_table<-pcl$feature_table
    sample_metadata<-pcl$sample_metadata
    feature_metadata<-pcl$feature_metadata

    ##########################
    # Loop over each method #
    #########################
    
    for (j in 1:length(base_learners)){
      
      #################
      # Print message #
      #################
      
      cat('Running method:', base_learners[j], '\n')
      
      ################################################
      # Per-feature method in a parallel environment #
      ################################################
      
      outputString<-paste('./Output/', datasets[i], '_', base_learners[j], '.RData', sep = '')
      
      if(!file.exists(outputString)){
        
        # Set Up Clustering Environment
        no_cores <- detectCores() - 6
        cl <- makeCluster(no_cores)
        registerDoParallel(cl)
        
        ##############################################################
        # Customize tuning parameters and run stacked generalization #  
        ##############################################################
        
        fit<-tryCatch(IntegratedLearner(feature_table,
                                                sample_metadata, 
                                                feature_metadata,
                                                feature_table_valid = feature_table_valid,
                                                sample_metadata_valid = sample_metadata_valid,
                                                folds = folds,
                                                base_learner = base_learners[j],
                                                meta_learner = "SL.nnls.auc",
                                                family=binomial(),
                                                run_stacked=FALSE),  error = function(err){NULL})
        
        
        # Stop the Cluster 
        stopCluster(cl)
        
        ######################
        # Save valid outputs #  
        #####################
        
        if(!is.null(fit)){
          save(fit, file = outputString)
        }
      }
    }
  }
}
