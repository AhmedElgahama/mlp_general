test_json_string_2 <-
  '[{
  "customerID" : 1,
  "gender" : "Male" ,
  "SeniorCitizen" : 0 ,
  "Partner" : "Yes" ,
  "Dependents" : "Yes" ,
  "tenure" : 70  ,
  "PhoneService" : "Yes" ,
  "MultipleLines" : "Yes",
  "InternetService" : "No" ,
  "OnlineSecurity" : "No internet service" ,
  "OnlineBackup" :  "No internet service" ,
  "DeviceProtection": "No internet service" ,
  "TechSupport" : "No internet service",
  "StreamingTV" : "No internet service",
  "StreamingMovies" : "No internet service",
  "Contract" : "Two year",
  "PaperlessBilling" : "No",
  "PaymentMethod" : "Mailed check",
  "MonthlyCharges" : 25.4,
  "TotalCharges" : 1782.05
},
{
  "customerID" : 1,
  "gender" : "Male" ,
  "SeniorCitizen" : 0 ,
  "Partner" : "Yes" ,
  "Dependents" : "Yes" ,
  "tenure" : 70  ,
  "PhoneService" : "Yes" ,
  "MultipleLines" : "Yes",
  "InternetService" : "No" ,
  "OnlineSecurity" : "No internet service" ,
  "OnlineBackup" :  "No internet service" ,
  "DeviceProtection": "No internet service" ,
  "TechSupport" : "No internet service",
  "StreamingTV" : "No internet service",
  "StreamingMovies" : "No internet service",
  "Contract" : "Two year",
  "PaperlessBilling" : "No",
  "PaymentMethod" : "Mailed check",
  "MonthlyCharges" : 25.4,
  "TotalCharges" : 1782.05
},
{
  "customerID" : 2,
  "gender" : "Male" ,
  "SeniorCitizen" : 0 ,
  "Partner" : "Yes" ,
  "Dependents" : "Yes" ,
  "tenure" : 70  ,
  "PhoneService" : "Yes" ,
  "MultipleLines" : "Yes",
  "InternetService" : "No" ,
  "OnlineSecurity" : "No internet service" ,
  "OnlineBackup" :  "No internet service" ,
  "DeviceProtection": "No internet service" ,
  "TechSupport" : "No internet service",
  "StreamingTV" : "No internet service",
  "StreamingMovies" : "No internet service",
  "Contract" : "Two year",
  "PaperlessBilling" : "No",
  "PaymentMethod" : "Mailed check",
  "MonthlyCharges" : 25.4,
  "TotalCharges" : 1782.05
}]'
  
## Example for testing
test_json_string <-
  '{
      "customerID" : 3,
      "gender" : "Male" ,
      "SeniorCitizen" : 0 ,
      "Partner" : "Yes" ,
      "Dependents" : "Yes" ,
      "tenure" : 70  ,
      "PhoneService" : "Yes" ,
      "MultipleLines" : "Yes",
      "InternetService" : "No" ,
      "OnlineSecurity" : "No internet service" ,
      "OnlineBackup" :  "No internet service" ,
      "DeviceProtection": "No internet service" ,
      "TechSupport" : "No internet service",
      "StreamingTV" : "No internet service",
      "StreamingMovies" : "No internet service",
      "Contract" : "Two year",
      "PaperlessBilling" : "No",
      "PaymentMethod" : "Mailed check",
      "MonthlyCharges" : 25.4,
      "TotalCharges" : 1782.05
}'


## ---- Initialising libraries ----
suppressWarnings(suppressMessages(library(tidyverse)))
suppressWarnings(suppressMessages(library(lubridate)))
suppressWarnings(suppressMessages(library(data.table)))
suppressWarnings(suppressMessages(library(httr)))
suppressWarnings(suppressMessages(library(glue)))

## Load model
trained_model  <- read_rds('/opt/model_inputs_outputs/model/artifacts/model.rds')
encode_these   <- trained_model$variables_to_encode
variables_numeric   <- trained_model$variables_numeric
target_class          <- trained_model$target_class
other_class           <- trained_model$other_class
id_column             <- trained_model$id_column
exp_vars              <- trained_model$exp_vars
threshold <- 0.50




prediction_scorer <- function(row) {
  ## Function to get data and return probability
  
  ## initialize scores
  score  <- 0
  
  ## Encode categorical features with number of training encoding
  encodings_tmp <-
    trained_model$encodings %>%
    map(function(x) {
      if (is.data.frame(x)) {
        x[, 2, drop = TRUE] %>% set_names(x[, 1, drop = TRUE])
      } else {
        x
      }
    })
  for (catvar in encode_these) {
    row[[catvar]] <-
      encodings_tmp[[catvar]][row[[catvar]] %>% as.character()]
  }
  
  ## Getting probability
  score <-
    predict(trained_model$mdl,
            row %>% dplyr::select(
              all_of(exp_vars)
            ))[,2]
  
  ## return prediction
  score
}





#* @post /infer
#* @serializer json list(auto_unbox=TRUE)
function(req) {
# pred <- function(req) {
  ## grab the request body 'req' and put it into the variable 'row'
  row <- jsonlite::fromJSON(req$postBody)$instances %>% as_tibble()
  # row <- jsonlite::fromJSON(req) %>% as_tibble()
  row %>% glimpse()
  
  ids <- row %>% select(id_column)
  
  print("$$$$$$$$$$$$$$$$$$$$$$")
  print(ids)
  print("$$$$$$$$$$$$$$$$$$$$$$")
  ## placeholder for JSON string to be printed at the end
  result <-
    tibble(
      predicted_class_prob = 0,
      predicted_class = '',
      warnings = ''
    )
  
  ## parameters that we need
  necessary_params <- trained_model$mdl$feature_names
  
  ## if we do NOT have all we need...
  if (!all(necessary_params %in% names(row))) {
    print(setdiff(necessary_params,names(row)))
    result$predicted_class_prob <- 0
    result$predicted_class <- ''
    result$warnings <-'Some necessary features are missing'# setdiff(necessary_params,names(row)) %>% as.character() #
    
  } else {
    ## keep only the necessary parameters
    row <- row[necessary_params]
    
    ## if any of the necessary parameters are null...
    if (row %>% sapply(is.null) %>% any()) {
      result$predicted_class_prob <- 0.55
      result$predicted_class <- 'UNCOVERED'
      result$warnings <-
        paste('The following required parameters were NULL:',
              null_parameters)
      
    } else {
      predicted_class_prob <- prediction_scorer(row)
      
      print(predicted_class_prob)
      
      # predicted_class <-
      #   if_else(predicted_class_prob >= 0.5 ,
      #           get("target_class"),
      #           get("other_class"))
      predicted_class <-
        if_else(predicted_class_prob >= 0.5 ,
                get("target_class"),
                get("other_class"))
      print(predicted_class)
      
      
      other_class <- if_else(predicted_class_prob >= 0.5 ,
                             get("other_class"),
                             get("target_class")
                             )
      print(other_class)
      
      # predicted_class_prob  <- 
      #   if_else(predicted_class_prob >= 0.5 ,
      #           predicted_class_prob %>% round(5),
      #           1 - predicted_class_prob %>% round(5))
    predicted_class_prob  <-
     if_else(predicted_class_prob >= 0.5 ,
                predicted_class_prob %>% round(5),
                1 - predicted_class_prob %>% round(5))
    print(predicted_class_prob)
    
    }
    
    probabilities_list <- list()
    
    for (i in 1:length(predicted_class_prob)) {
      
      l = list()
      
      p_c <- predicted_class[[i]] 
      o_c <- other_class[[i]]
      
      p_c <- p_c %>% as.character()
      o_c <- o_c %>% as.character()
      
      p_p <- predicted_class_prob[[i]] %>% as.numeric()
      o_p <- 1 - predicted_class_prob[[i]] %>% as.numeric()
      
      
      l[[p_c]] = p_p
      l[[o_c]] = o_p
      
      lst_obj <- list(c1 = p_p,
                      c2 = o_p)
      names(lst_obj) <- c(get("p_c"),get("o_c"))
      
      probabilities_list[[length(probabilities_list)+1]] <- lst_obj
    }
    
    
    
    
    
    
    l <- list() 
    
    for(i in 1:length(probabilities_list)){
      
      
      instance <- list(ids[[i,1]],predicted_class[[i]],probabilities_list[[i]])
      names(instance) <- c(id_column, "label", "probabilities")
      
      
      l[[i]] <- instance
    }
    
    
    
    m = list()
    
    m[[1]] <- l
    
    names(m) <- c( "predictions")
    

    h <- m
    
    h

}
}




#* @get /ping
#* @serializer json list(auto_unbox=TRUE)
endpoint.healthz <- function(req) {
  return("it's working perfectly")
}
