

library(tidyverse)
library(lubridate)
library(data.table)
library(dtplyr)
library(tictoc)
library(glue)
library(pROC)
library(MASS) 
library(neuralnet)
options(dplyr.summarise.inform = FALSE)


trainer_func <- function(train_set, 
                         validation_set, 
                         explanatory_variables, 
                         target_variable,
                         hypergrid,
                         target_variable_mapping = NULL) 
{
  
  n <- explanatory_variables
  f <- as.formula(paste("label ~",  
                        paste(n[!n %in% "label"], 
                              collapse = " + "))) 
  
  neural_net2 = neuralnet( f , data = train_set %>% dplyr::select(explanatory_variables , label), 
                          hidden =  1, linear.output = TRUE,threshold = 0.01, stepmax=1000000) 
  
  
  
  val_features   <- validation_set %>% dplyr::select(all_of(explanatory_variables))
  
  val_predictions <- predict(neural_net2, val_features)[,1]
  val_predictions <- val_predictions >0.5
  val_predictions <- val_predictions %>% as.numeric()
  val_labels     <- validation_set[[target_variable]] %>% as.numeric()
  score <- auc(roc(val_labels, val_predictions)) %>% as.numeric()
  print(score)
  
  
  stuff <- list()
  stuff$mdl <- neural_net2
  stuff$score <- score
  
  return(stuff)
}







tester_func <- function(mdl, test_set,exp_vars) {
  
  test_features <- test_set %>% dplyr::select(all_of(exp_vars))
  test_predictions <- predict(mdl, test_features)
  test_predictions <- test_predictions >0.5
  test_predictions <- test_predictions %>% as.numeric()
  results <- list()
  results[['test_predictions']] <- 
    tibble(pred = test_predictions#, 
           # actual = test_labels
           ) #%>% 
    # mutate(actual_aux = test_set[[target_variable]] %>% as.character() %>% paste(actual))
  
  results
  
}


## *************************

my_personal_encoder <- function(df1, df2 = NULL, encode_these, target_variable_mapping) {
  
  df1 <- df1 %>% mutate(y = target_variable_mapping[label] %>% as.logical())
  if (!is.null(df2)) {
    df2 <- df2 %>% mutate(y = target_variable_mapping[label] %>% as.logical())
  }
  
  encodings <- list()
  for(v in encode_these) {
    if (!is.null(df2)) {
      c(df1,   df2, garbage, encodings[[v]]) %<-% general_purpose_encoder(df1, df2, df2, v, 'target')
    } else {
      c(df1, dummy, garbage, encodings[[v]]) %<-% general_purpose_encoder(df1, df1, df1, v, 'target')
    }
  }
  
  df1 <- df1 %>% dplyr::select(-y)
  df1 <- df1 %>% filter_if(is.numeric, ~ !is.infinite(.))
  colnames(df1) <- colnames(df1) %>% str_replace('_target_encoded$', '')
  
  if (!is.null(df2)) {
    df2 <- df2 %>% dplyr::select(-y)
    df2 <- df2 %>% filter_if(is.numeric, ~ !is.infinite(.))
    colnames(df2) <- colnames(df2) %>% str_replace('_target_encoded$', '')
    return(list(df1, df2, encodings))
    
  } else {
    return(list(df1, df1, encodings))
  }
  
}




general_purpose_encoder <- function(training,
                                    validation,
                                    test,
                                    column,
                                    encoding) {
  if (encoding == 'target') {
    c(training, validation, test, encodings) %<-% target_encode_category(training, validation, test, column)
  }
  if (encoding == 'woe') {
    c(training, validation, test, encodings) %<-% woe_encode_category(training, validation, test, column)
  }
  if (encoding == 'ohe') {
    c(training, validation, test, encodings) %<-% ohe_encode_category(training, validation, test, column)
  }
  if (encoding == 'binary') {
    c(training, validation, test, encodings) %<-% binary_encode_category(training, validation, test, column)
  }
  return(list(training, validation, test, encodings))
}







target_encoder <- function(df, column, smooth_weight = 0) {
  global_rate <-
    df %>% count(y) %>% mutate(pct = n / sum(n)) %>% filter(!y) %>% pull(pct)
  
  target_encodings <- df %>%
    get_rejection_rates_by_category(column) %>%
    mutate(global_mean = global_rate) %>%
    mutate(
      smoothed_mean = (
        `Total applicants` * `Rejection rate` + smooth_weight * global_mean
      ) / (`Total applicants` + smooth_weight)
    ) %>%
    dplyr::select(Categories, smoothed_mean) %>% drop_na(Categories)
  
  
  return(target_encodings)
}


target_encode_category <- function(training, validation, test, column) {
  job_target_encoded <- training %>%
    target_encoder(column) %>%
    mutate(smoothed_mean = 1 - smoothed_mean) %>%
    set_names(c(column, glue('{column}_target_encoded')))
  
  training <- training %>%
    left_join(job_target_encoded) %>%
    dplyr::select(-column)
  validation <- validation %>%
    left_join(job_target_encoded) %>%
    dplyr::select(-column)
  test <- test %>%
    left_join(job_target_encoded) %>%
    dplyr::select(-column)
  
  return(list(training, validation, test, job_target_encoded))
}

target_encode_these <-
  function(df_train_acc_rej,
           df_val_acc_rej,
           variables_to_encode) {
    encodings <- list()
    for (my_var in variables_to_encode) {
      c(df_train_acc_rej,
        df_val_acc_rej,
        dummmy_var,
        encodings[[my_var]]) %<-% general_purpose_encoder(df_train_acc_rej,
                                                          df_val_acc_rej,
                                                          df_val_acc_rej,
                                                          my_var,
                                                          'target')
    }
    return(list(
      df_train_acc_rej,
      df_val_acc_rej,
      dummmy_var,
      encodings
    ))
  }


# Expects y to be TRUE/FALSE
get_rejection_rates_by_category <- function(df, col_name) {
  df %>%
    mutate(y = if_else(y, 'Y', 'N')) %>%
    count(!!sym(col_name), y) %>%
    group_by(!!sym(col_name)) %>%
    mutate(pct = n / sum(n), total_applicants = sum(n)) %>%
    ungroup() %>%
    dplyr::select(-n) %>%
    pivot_wider(names_from = y, values_from = pct) %>%
    replace_na(list(Y = 0, N = 0)) %>%
    dplyr::select(all_of(col_name), total_applicants, N) %>%
    set_names(c('Categories', 'Total applicants', 'Rejection rate')) %>%
    mutate(Column = col_name) %>%
    dplyr::select(Column, everything())
}






calc_mode <- function(x){
  
  # List the distinct / unique values
  distinct_values <- unique(x)
  
  # Count the occurrence of each distinct value
  distinct_tabulate <- tabulate(match(x, distinct_values))
  
  # Return the value with the highest occurrence
  distinct_values[which.max(distinct_tabulate)]
}

