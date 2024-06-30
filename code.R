library(MASS)
library(caret)
library(e1071)
library(Metrics)
library(lqr)
library(pROC)
library(progress)
library(xtable)
library(ggplot2)
library(yardstick)
library(MLmetrics)
library(corrplot)
set.seed(123)

# Helper Functions --------------------------------------------------------
save_latex <- function(results, exp_name) {
  paths <- get_paths(exp_name)
  latex_table <- xtable(results, type = "latex", display = c("fg", "fg", "fg", "fg", "fg", "fg", "fg"))
  print(latex_table, file = paths$latex_table, digits = 5)
}

sigmoid <- function(z) 1 / (1 + exp(-z))

compute_means_and_stds <- function(all_results, quantiles) {
  compute_stats <- function(metrics_list) {
    valid_metrics <- Filter(Negate(is.null), metrics_list)
    if (length(valid_metrics) > 0) {
      metrics_matrix <- do.call(rbind, valid_metrics)
      means <- colMeans(metrics_matrix, na.rm = TRUE)
      stds <- apply(metrics_matrix, 2, sd, na.rm = TRUE)
      result <- data.frame(mean = round(means, 5), std = round(stds, 5))
      combined <- apply(result, 1, function(x) paste0(x['mean'], " (", x['std'], ")"))
      names(combined) <- colnames(metrics_matrix)
      return(combined)
    } else {
      return(rep(NA, length(metrics_list[[1]])))
    }
  }
  
  log_stats <- if (length(all_results$log_metrics) > 0) {
    compute_stats(all_results$log_metrics)
  } else {
    rep(NA, length(all_results$log_metrics[[1]]))
  }
  
  svm_stats <- if (length(all_results$svm_metrics) > 0) {
    compute_stats(all_results$svm_metrics)
  } else {
    rep(NA, length(all_results$svm_metrics[[1]]))
  }
  
  qlr_stats <- list()
  
  for (q in quantiles) {
    q_name <- paste0("qlr_metrics_", q)
    if (length(all_results[[q_name]]) > 0) {
      qlr_stats[[q_name]] <- compute_stats(all_results[[q_name]])
    } else {
      qlr_stats[[q_name]] <- rep(NA, length(all_results[[q_name]][[1]]))
    }
  }
  
  log_stats_df <- data.frame(model = "Logistic Regression", t(log_stats))
  svm_stats_df <- data.frame(model = "SVM", t(svm_stats))
  qlr_stats_dfs <- do.call(rbind, lapply(names(qlr_stats), function(q) {
    metrics <- qlr_stats[[q]]
    if (!is.null(metrics)) {
      return(data.frame(model = q, t(metrics)))
    }
  }))
  
  all_stats_results <- rbind(log_stats_df, svm_stats_df, qlr_stats_dfs)
  return(all_stats_results)
}

get_paths <- function(exp_name) {
  exp_dir <- file.path("/Users/amar/Desktop/Thesis")
  list(
    roc_plot = file.path(exp_dir, paste0("roc_curves_", exp_name, ".pdf")),
    latex_table = file.path(exp_dir, paste0("all_results_", exp_name, ".tex"))
  )
}


# Modelling ---------------------------------------------------------------

modelling <- function(data, p = 5, quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), roc = FALSE, exp_name = "experiment") {
  old_warning_setting <- options(warn = -1)
  
  # Suppress messages and output by redirecting to a temporary file
  temp_file <- tempfile()
  output_con <- file(temp_file, open = "wt")
  message_con <- file(temp_file, open = "wt")
  sink(output_con)
  sink(message_con, type = "message")
  
  on.exit({
    # Restore output and warning settings on exit
    sink(type = "message")
    sink()
    close(output_con)
    close(message_con)
    unlink(temp_file)
    options(old_warning_setting)
  })
  
  paths <- get_paths(exp_name)
  train_index <- createDataPartition(data$Y, p = 0.7, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  results_list <- list()
  coefficients <- list()
  
  ## Logistic Regression
  log_model <- glm(Y ~ ., family = binomial, data = train_data)
  coefficients$log_coeff <- log_model$coefficients
  log_test_probabilities <- predict(log_model, newdata = test_data, type = "response")
  log_test_predictions <- ifelse(log_test_probabilities > 0.5, 1, 0)
  recall_value <- Metrics::recall(log_test_predictions, test_data$Y)
  precision_value <- Metrics::precision(log_test_predictions, test_data$Y)
  roc_log <- suppressWarnings({roc(test_data$Y, log_test_probabilities, levels = c(0, 1), direction = "<")})
  auc_log <- auc(roc_log)
  log_metrics <- c(
    accuracy = Metrics::accuracy(log_test_predictions, test_data$Y),
    recall_value = recall_value,
    precision_value = precision_value,
    f1 = 2 * precision_value * recall_value / (precision_value + recall_value),
    auc = auc_log
  )
  results_list$log_metrics <- log_metrics
  
  ## SVM
  svm_model <- svm(Y ~ ., data = train_data, probability = TRUE)
  svm_test_probabilities <- predict(svm_model, newdata = test_data, probability = TRUE)
  svm_test_predictions <- ifelse(svm_test_probabilities > 0.5, 1, 0)
  svm_coefficients <- t(svm_model$coefs) %*% svm_model$SV
  svm_intercept <- -svm_model$rho
  coefficients$svm_coeff <- c(svm_coefficients, svm_intercept)
  recall_value <- Metrics::recall(svm_test_predictions, test_data$Y)
  precision_value <- Metrics::precision(svm_test_predictions, test_data$Y)
  roc_svm <- suppressWarnings({roc(test_data$Y, svm_test_probabilities, levels = c(0, 1), direction = "<")})
  auc_svm <- auc(roc_svm)
  svm_metrics <- c(
    accuracy = Metrics::accuracy(svm_test_predictions, test_data$Y),
    recall_value = recall_value,
    precision_value = precision_value,
    f1 = 2 * precision_value * recall_value / (precision_value + recall_value),
    auc = auc_svm
  )
  results_list$svm_metrics <- svm_metrics
  
  ## Quantile Logistic Regression for specified quantiles
  quantiles <- quant
  qlr_metrics_list <- list()
  
  if (roc) {
    pdf(paths$roc_plot)
    plot(roc_log, col = "red", main = "ROC Curves")
    legend_labels <- c("Logistic Regression")
    legend_colors <- c("red")
    lines(roc_svm, col = "blue")
    legend_labels <- c(legend_labels, "SVM")
    legend_colors <- c(legend_colors, "blue")
  }
  
  for (q in quantiles) {
    tryCatch(
      {
        lqr_model <- Log.lqr(Y ~ ., data = train_data, p = q, dist = "t", silent = TRUE)
        coeff <- lqr_model$beta
        coefficients[[paste0("QLR_", q)]] <- coeff
        predictor_matrix <- model.matrix(~., data = test_data)
        linear_predictor <- predictor_matrix[, c(1, 3:(p + 2))] %*% coeff
        qlr_test_probabilities <- sigmoid(linear_predictor)
        qlr_test_predictions <- ifelse(qlr_test_probabilities > q, 1, 0)
        recall_value <- Metrics::recall(qlr_test_predictions, test_data$Y)
        precision_value <- Metrics::precision(qlr_test_predictions, test_data$Y)
        roc_qlr <- suppressWarnings({roc(test_data$Y, qlr_test_probabilities, levels = c(0, 1), direction = "<")})
        if (roc) {
          lines(roc_qlr, col = rainbow(length(quantiles))[q * 10])
          legend_labels <- c(legend_labels, paste("QLR", q))
          legend_colors <- c(legend_colors, rainbow(length(quantiles))[q * 10])
        }
        auc_qlr <- auc(test_data$Y, qlr_test_probabilities)
        qlr_metrics <- c(
          accuracy = Metrics::accuracy(qlr_test_predictions, test_data$Y),
          recall_value = recall_value,
          precision_value = precision_value,
          f1 = 2 * precision_value * recall_value / (precision_value + recall_value),
          auc = auc_qlr
        )
        qlr_metrics_list[[paste0("QLR_", q)]] <- qlr_metrics
      },
      error = function(e) {
        message(paste("Error at quantile:", q, "->", e$message))
        qlr_metrics_list[[paste0("QLR_", q)]] <- NA
      }
    )
  }
  if (roc) {
    legend("bottomright", legend = legend_labels, col = legend_colors, lwd = 2)
    dev.off()
  }
  results_list$qlr_metrics <- qlr_metrics_list
  
  # Combine all results
  log_results <- data.frame(model = "Logistic Regression", t(log_metrics))
  svm_results <- data.frame(model = "SVM", t(svm_metrics))
  qlr_results_combined <- do.call(rbind, lapply(names(qlr_metrics_list), function(q) {
    data.frame(model = q, t(qlr_metrics_list[[q]]))
  }))
  
  all_results <- rbind(log_results, svm_results, qlr_results_combined)
  
  return(list(all_results = all_results, results_list = results_list, coefficients = coefficients))
}


# Experiments Initialization ----------------------------------------------
log_gen_data <- function(n = 1000, p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), inter = FALSE, higher = FALSE) {
  X <- cbind(1, matrix(rnorm(n * p), n, p))
  X_generate <- X
  
  # Add interaction terms if inter is TRUE
  if (inter) {
    interaction1 <- X[, 2] * X[, 3]
    interaction2 <- X[, 3] * X[, 4]
    X_generate <- cbind(X, interaction1, interaction2)
  }
  
  # Add higher-order terms if higher is TRUE
  if (higher) {
    interaction1 <- X[, 2]^2
    interaction2 <- X[, 3]^2
    X_generate <- cbind(X, interaction1, interaction2)
  }
  
  if (length(theta) < ncol(X_generate)) {
    additional_values <- c(0.5, 0.8, 0.4, 0.3, 0.7, 0.9)
    additional_values_needed <- ncol(X_generate) - length(theta)
    additional_values_repeated <- rep(additional_values, length.out = additional_values_needed)
    theta <- c(theta, additional_values_repeated)
  }
  
  linear_predictor <- X_generate %*% theta
  sigmoid <- function(z) 1 / (1 + exp(-z))
  probabilities <- sigmoid(linear_predictor)
  Y <- rbinom(n, 1, probabilities)
  data <- data.frame(Y, X[, -1])
  
  return(list(data = data, theta = theta))
}

run <- function(sample_sizes = c(1000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", concat = TRUE, inter = FALSE, higher = FALSE, mean_diff = c(1, 0), noise_level = 0, reps = NULL, roc = FALSE) {
  results_all_n <- list()
  coefficients_all_n <- list()
  total_iterations <- if (is.null(reps)) length(sample_sizes) else length(sample_sizes) * reps
  pb <- progress_bar$new(
    format = "  Running [:bar] :percent eta: :eta",
    total = total_iterations,
    clear = FALSE,
    width = 60
  )
  
  start_time <- Sys.time()
  iteration <- 1
  
  for (n in sample_sizes) {
    if (is.null(reps)) {
      if (method == "log") {
        generated <- log_gen_data(n, p, theta, inter, higher)
      } else if (method == "svm") {
        generated <- svm_gen_data(n, p, concat = concat, mean_diff = mean_diff, noise_level = noise_level)
      } else {
        stop("Invalid method. Use 'log' or 'svm'.")
      }
      data <- generated$data
      results <- modelling(data, p, quant, exp_name = "exp", roc = roc)
      results_all_n[[paste("exp", n, sep = "_")]] <- results$all_results
      coefficients_all_n[[paste("exp", n, sep = "_")]] <- results$coefficients
      pb$tick()
    } else {
      all_results <- list(log_metrics = list(), svm_metrics = list())
      all_coefficients <- list(log_coeff = list())
      
      for (q in quant) {
        all_results[[paste0("qlr_metrics_", q)]][[1]] <- list()
        all_coefficients[[paste0("QLR_", q)]][[1]] <- list()
      }
      
      for (x in 1:reps) {
        if (method == "log") {
          generated <- log_gen_data(n, p, theta, inter)
        } else if (method == "svm") {
          generated <- svm_gen_data(n, p)
        } else {
          stop("Invalid method. Use 'log' or 'svm'.")
        }
        data <- generated$data
        results <- modelling(data, p, quant, exp_name = "exp1", roc = roc)
        iteration_results <- results$results_list
        iteration_coefficients <- results$coefficients
        
        all_results$log_metrics[[x]] <- iteration_results$log_metrics
        all_results$svm_metrics[[x]] <- iteration_results$svm_metrics
        all_coefficients$log_coeff[[x]] <- iteration_coefficients$log_coeff
        
        for (q in quant) {
          q_name <- paste0("QLR_", q)
          all_results[[paste0("qlr_metrics_", q)]][[x]] <- iteration_results$qlr_metrics[[q_name]]
          all_coefficients[[paste0("QLR_", q)]][[x]] <- iteration_coefficients[[q_name]]
        }
        pb$tick()
      }
      
      mean_results <- compute_means_and_stds(all_results, quant)
      results_all_n[[paste("exp", n, sep = "_")]] <- mean_results
      
      mean_log_coeff <- rowMeans(do.call(cbind, all_coefficients$log_coeff))
      mean_qlr_coeff <- lapply(quant, function(q) {
        q_name <- paste0("QLR_", q)
        rowMeans(do.call(cbind, all_coefficients[[q_name]]))
      })
      names(mean_qlr_coeff) <- paste0("QLR_", quant)
      
      coefficients_all_n[[paste("exp", n, sep = "_")]] <- list(log_coeff = mean_log_coeff, qlr_coeff = mean_qlr_coeff)
    }
    
    iteration <- iteration + 1
  }
  
  end_time <- Sys.time()
  total_time <- difftime(end_time, start_time, units = "secs")
  print(paste("Total time:", total_time))
  sink()
  
  return(list(results_all_n = results_all_n, coefficients_all_n = coefficients_all_n))
}

# Experiment 1 ------------------------------------------------------------

exp_1_100 <- run(sample_sizes = c(100), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, NULL)
exp_1_100$results_all_n

exp_1_500 <- run(sample_sizes = c(500), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, NULL)
exp_1_500$results_all_n

exp_1_1000 <- run(sample_sizes = c(1000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, NULL)
exp_1_1000$results_all_n

exp_1_5000 <- run(sample_sizes = c(5000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, NULL)
exp_1_5000$results_all_n


# Experiment 2 ------------------------------------------------------------
exp_2_100 <- run(sample_sizes = c(100), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_2_100$results_all_n

exp_2_500 <- run(sample_sizes = c(500), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_2_500$results_all_n

exp_2_1000 <- run(sample_sizes = c(1000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_2_1000$results_all_n

exp_2_5000 <- run(sample_sizes = c(5000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_2_5000$results_all_n

exp_2_10000 <- run(sample_sizes = c(10000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_2_10000$results_all_n


# Experiment 3 ------------------------------------------------------------

exp_3_500_p5 <- run(sample_sizes = c(500), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_3_500_p5$results_all_n

exp_3_500_p10 <- run(sample_sizes = c(500), p = 10, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_3_500_p10$results_all_n

exp_3_500_p15 <- run(sample_sizes = c(500), p = 15, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_3_500_p15$results_all_n

exp_3_500_p20 <- run(sample_sizes = c(500), p = 20, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_3_500_p20$results_all_n

exp_3_500_p50 <- run(sample_sizes = c(500), p = 50, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = F, reps = c(1000))
exp_3_500_p50$results_all_n

# Experiment 4 ------------------------------------------------------------

exp_4_100 <- run(sample_sizes = c(100), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = T, higher = T, roc = F, reps = c(1000))
exp_4_100$results_all_n

exp_4_500 <- run(sample_sizes = c(500), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = T, higher = T, roc = F, reps = c(1000))
exp_4_500$results_all_n

exp_4_1000 <- run(sample_sizes = c(1000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = T, higher = T, roc = F, reps = c(1000))
exp_4_1000$results_all_n

exp_4_5000 <- run(sample_sizes = c(5000), p = 5, theta = c(-1, 0.5, -0.3, 0.5, -0.6, 0.7), quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), method = "log", inter = T, higher = T, roc = F, reps = c(1000))
exp_4_5000$results_all_n


# REAL LIFE: HOTEL --------------------------------------------------------
hotel_data <- read.csv('/Users/amar/Downloads/hotel.csv')
names(hotel_data)[names(hotel_data) == "is_canceled"] <- "Y"
print(names(hotel_data))
sapply(hotel_data, function(x) if(is.factor(x) | is.character(x)) unique(x))

# Recode variables
hotel_data$hotel <- ifelse(hotel_data$hotel == "Resort Hotel", 0, 1)
hotel_data$meal <- ifelse(hotel_data$meal == "BB", 0, 
                          ifelse(hotel_data$meal == "FB", 1,
                                 ifelse(hotel_data$meal == "HB", 2,
                                        ifelse(hotel_data$meal == "SC", 3, 4))))

hotel_data$market_segment <- ifelse(hotel_data$market_segment == "Direct", 0, 
                                    ifelse(hotel_data$market_segment == "Corporate", 1,
                                           ifelse(hotel_data$market_segment == "Online TA", 2,
                                                  ifelse(hotel_data$market_segment == "Offline TA/TO", 3,
                                                         ifelse(hotel_data$market_segment == "Complementary", 4,
                                                                ifelse(hotel_data$market_segment == "Groups", 5,
                                                                       ifelse(hotel_data$market_segment == "Undefined", 6, 7)))))))

hotel_data$distribution_channel <- ifelse(hotel_data$distribution_channel == "Direct", 0, 
                                          ifelse(hotel_data$distribution_channel == "Corporate", 1,
                                                 ifelse(hotel_data$distribution_channel == "TA/TO", 2,
                                                        ifelse(hotel_data$distribution_channel == "Undefined", 3, 4))))

hotel_data$deposit_type <- ifelse(hotel_data$deposit_type == "No Deposit", 0, 
                                  ifelse(hotel_data$deposit_type == "Refundable", 1, 2))

hotel_data$customer_type <- ifelse(hotel_data$customer_type == "Transient", 0, 
                                   ifelse(hotel_data$customer_type == "Contract", 1,
                                          ifelse(hotel_data$customer_type == "Transient-Party", 2, 3)))

hotel_data$reservation_status <- ifelse(hotel_data$reservation_status == "Check-Out", 0, 
                                        ifelse(hotel_data$reservation_status == "Canceled", 1, 2))

# Convert relevant columns to numeric
numeric_columns <- c("lead_time", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
                     "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", 
                     "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", 
                     "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", 
                     "total_of_special_requests")

hotel_data[numeric_columns] <- lapply(hotel_data[numeric_columns], as.numeric)

# Convert reservation_status_date to Date type
hotel_data$reservation_status_date <- as.Date(hotel_data$reservation_status_date)

# Select variables for correlation analysis
variables_to_include <- c("Y", "hotel", "meal", "market_segment", "distribution_channel", 
                          "deposit_type", "customer_type", "lead_time", 
                          "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
                          "stays_in_weekend_nights", "stays_in_week_nights", "adults", 
                          "children", "babies", "is_repeated_guest", 
                          "previous_cancellations", "previous_bookings_not_canceled", 
                          "booking_changes", "days_in_waiting_list", "adr", 
                          "required_car_parking_spaces", "total_of_special_requests")

# Subset dataset
hotel_data_subset <- hotel_data[, variables_to_include]

# Calculate correlation matrix
correlation_matrix <- cor(hotel_data_subset, use = "complete.obs")

# Extract and order correlations with Y
correlations_with_Y <- correlation_matrix["Y", ]
ordered_correlations <- sort(abs(correlations_with_Y), decreasing = TRUE)
ordered_correlations

top_10_variables <- c("Y", "deposit_type", "lead_time", "market_segment", "total_of_special_requests", 
                      "required_car_parking_spaces", "distribution_channel", "booking_changes", 
                      "hotel", "customer_type")

# Subset dataset to top 10 variables
hotel_data_top_10 <- hotel_data[, top_10_variables]

# Calculate correlation matrix for top 10 variables
correlation_matrix_top_10 <- cor(hotel_data_top_10, use = "complete.obs")

# Plot the correlation matrix
corrplot(correlation_matrix_top_10, method = "circle", type = "upper", 
         tl.col = "black", tl.cex = 0.6, addCoef.col = "black", number.cex = 0.6)

# Create a bar plot to check the balance of Y
ggplot(hotel_data, aes(x = factor(Y))) +
  geom_bar(fill = "steelblue") +
  xlab("Booking Cancellation") +
  ylab("Count") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 10)
  )

# Subset new hotel data for modelling
new_hotel_data <- hotel_data[1:5000, top_10_variables[1:6]]

hotel_results <- modelling(new_hotel_data, p = 2, quant = c(0.6, 0.35), roc = F, exp_name = "hotel_experiment")
hotel_results$all_results
hotel_results$coefficients

# REAL LIFE: HEART --------------------------------------------------------------------
heart_data <- read.csv('/Users/amar/Downloads/heart.csv')

# Rename and recode the data
names(heart_data)[names(heart_data) == "HeartDisease"] <- "Y"

# Recode the other categorical variables
heart_data$ChestPainType <- ifelse(heart_data$ChestPainType == "ATA", 0, 
                                   ifelse(heart_data$ChestPainType == "NAP", 1, 2))
heart_data$RestingECG <- ifelse(heart_data$RestingECG == "Normal", 0, 1)
heart_data$ExerciseAngina <- ifelse(heart_data$ExerciseAngina == "N", 0, 1)
heart_data$ST_Slope <- ifelse(heart_data$ST_Slope == "Up", 0, 
                              ifelse(heart_data$ST_Slope == "Flat", 1, 2))
heart_data$RestingBP <- as.numeric(heart_data$RestingBP)
heart_data$Cholesterol <- as.numeric(heart_data$Cholesterol)
heart_data$FastingBS <- as.numeric(heart_data$FastingBS)
heart_data$MaxHR <- as.numeric(heart_data$MaxHR)
heart_data$Oldpeak <- as.numeric(heart_data$Oldpeak)
heart_data$Sex <- ifelse(heart_data$Sex == "M", 0, 1)

# Create a bar plot to check the balance of Y
ggplot(heart_data, aes(x = factor(Y))) +
  geom_bar(fill = "steelblue") +
  xlab("Heart Disease") +
  ylab("Count") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    axis.text.x = element_text(size = 13),
    axis.text.y = element_text(size = 10)
  )

# Calculate the correlation matrix
correlation_matrix <- cor(heart_data, use = "complete.obs")

# Extract and order correlations with Y
correlations_with_Y <- correlation_matrix["Y", ]
ordered_correlations <- sort(abs(correlations_with_Y), decreasing = TRUE)
ordered_correlations

# Plot the correlation matrix
corrplot(correlation_matrix, method = "color", type = "full", tl.cex = 0.8, title = "", addCoef.col = "black", number.cex = 0.7)

# Subset data for modelling with exercise test
heart_new_data <- heart_data[, c("Y", "ST_Slope", "ChestPainType", "ExerciseAngina", "Oldpeak", "Sex", "Age")]
heart_new_data <- na.omit(heart_new_data)
heart_results <- modelling(heart_new_data, p = 6, quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), roc = FALSE, exp_name = "heart_experiment")
heart_results$all_results

# Subset data for modelling at home
heart_new_data <- heart_data[, c("Y", "ChestPainType", "Sex", "Age", "Cholesterol")]
heart_new_data <- na.omit(heart_new_data)
heart_results <- modelling(heart_new_data, p = 4, quant = c(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), roc = FALSE, exp_name = "heart_experiment")
heart_results$all_results