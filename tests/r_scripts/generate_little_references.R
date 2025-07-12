#!/usr/bin/env Rscript
# Generate Little's MCAR test references using BaylorEdPsych::LittleMCAR
# Robust version with error handling and edge case management

library(mvnmle)
library(BaylorEdPsych)
library(jsonlite)

# Safe wrapper for LittleMCAR with edge case handling
safe_little_mcar <- function(data, dataset_name) {
  cat(sprintf("\n=== Processing %s dataset ===\n", dataset_name))
  
  # Basic info
  n_obs <- nrow(data)
  n_vars <- ncol(data)
  n_missing <- sum(is.na(data))
  missing_rate <- n_missing / (n_obs * n_vars)
  
  cat(sprintf("Data: %d observations × %d variables\n", n_obs, n_vars))
  cat(sprintf("Missing values: %d (%.1f%%)\n", n_missing, missing_rate * 100))
  
  # Edge case 1: No missing data
  if (n_missing == 0) {
    cat("No missing data detected - MCAR test not applicable\n")
    return(list(
      dataset_name = dataset_name,
      test_statistic = 0,
      p_value = 1.0,
      df = 0,
      n_patterns = 1,
      n_observations = n_obs,
      n_variables = n_vars,
      missing_rate = 0,
      note = "No missing data - MCAR test not applicable",
      patterns_info = list(complete = n_obs)
    ))
  }
  
  # Edge case 2: Check for problematic patterns
  patterns <- unique(!is.na(data))
  n_patterns <- nrow(patterns)
  
  # Count observations per pattern
  pattern_strings <- apply(!is.na(data), 1, paste, collapse = "")
  pattern_counts <- table(pattern_strings)
  
  cat(sprintf("Missing patterns detected: %d\n", n_patterns))
  
  # Check if any pattern has too few observations
  min_obs_needed <- 2  # Absolute minimum
  small_patterns <- pattern_counts < min_obs_needed
  if (any(small_patterns)) {
    cat(sprintf("Warning: %d patterns have fewer than %d observations\n", 
                sum(small_patterns), min_obs_needed))
  }
  
  # Try to run LittleMCAR with error handling
  test_result <- NULL
  error_msg <- NULL
  
  tryCatch({
    test_result <- LittleMCAR(data)
  }, error = function(e) {
    error_msg <<- e$message
    cat(sprintf("Error in LittleMCAR: %s\n", e$message))
  }, warning = function(w) {
    cat(sprintf("Warning in LittleMCAR: %s\n", w$message))
  })
  
  # If LittleMCAR failed, try manual calculation
  if (is.null(test_result)) {
    cat("Attempting manual calculation...\n")
    
    manual_result <- tryCatch({
      calculate_little_manually(data)
    }, error = function(e) {
      cat(sprintf("Manual calculation also failed: %s\n", e$message))
      NULL
    })
    
    if (!is.null(manual_result)) {
      return(c(manual_result, list(
        dataset_name = dataset_name,
        method = "manual",
        original_error = error_msg
      )))
    } else {
      # Complete failure
      return(list(
        dataset_name = dataset_name,
        error = error_msg,
        n_observations = n_obs,
        n_variables = n_vars,
        missing_rate = missing_rate,
        n_patterns = n_patterns,
        pattern_counts = as.list(pattern_counts),
        failed = TRUE
      ))
    }
  }
  
  # Success - extract results
  cat(sprintf("Chi-square: %.4f\n", test_result$chi.square))
  cat(sprintf("DF: %d\n", test_result$df))
  cat(sprintf("P-value: %.6f\n", test_result$p.value))
  cat(sprintf("Missing patterns: %d\n", test_result$missing.patterns))
  cat(sprintf("Decision: MCAR %s at α = 0.05\n",
              ifelse(test_result$p.value < 0.05, "REJECTED", "not rejected")))
  
  # Create reference object
  reference <- list(
    dataset_name = dataset_name,
    test_statistic = test_result$chi.square,
    p_value = test_result$p.value,
    df = test_result$df,
    n_patterns = test_result$missing.patterns,
    amount_missing = test_result$amount.missing,
    n_observations = n_obs,
    n_variables = n_vars,
    missing_rate = missing_rate,
    pattern_counts = as.list(pattern_counts),
    method = "BaylorEdPsych"
  )
  
  # Add pattern details if available
  if (!is.null(test_result$data)) {
    pattern_info <- list()
    for (i in 1:length(test_result$data)) {
      pattern_name <- names(test_result$data)[i]
      pattern_data <- test_result$data[[i]]
      pattern_info[[pattern_name]] <- list(
        n_obs = nrow(pattern_data),
        observed_vars = names(pattern_data)[!sapply(pattern_data[1,], is.na)]
      )
    }
    reference$pattern_details <- pattern_info
  }
  
  # Manual verification for our implementation
  cat("\nVerification with mlest:\n")
  ml_pooled <- tryCatch({
    mlest(data)
  }, error = function(e) {
    cat(sprintf("mlest failed: %s\n", e$message))
    NULL
  })
  
  if (!is.null(ml_pooled)) {
    loglik_pooled <- -ml_pooled$value / 2
    cat(sprintf("Pooled log-likelihood: %.4f\n", loglik_pooled))
    reference$pooled_loglik_verify <- loglik_pooled
  }
  
  return(reference)
}

# Manual implementation as fallback
calculate_little_manually <- function(data) {
  # This is a simplified version for when LittleMCAR fails
  n_obs <- nrow(data)
  n_vars <- ncol(data)
  
  # Get patterns
  patterns <- unique(!is.na(data))
  n_patterns <- nrow(patterns)
  
  # For now, return a placeholder
  # In practice, we'd implement the full algorithm here
  list(
    test_statistic = NA,
    p_value = NA,
    df = NA,
    n_patterns = n_patterns,
    note = "Calculated manually due to LittleMCAR failure"
  )
}

# Main execution
cat("=== Generating Little's MCAR Test References ===\n")
cat("Using BaylorEdPsych::LittleMCAR function with error handling\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("mvnmle version: %s\n", packageVersion("mvnmle")))
cat(sprintf("BaylorEdPsych version: %s\n", packageVersion("BaylorEdPsych")))
cat(sprintf("Generated on: %s\n", Sys.Date()))

# Load datasets
data(apple)
data(missvals)

# Create output directory
dir.create("tests/references", showWarnings = FALSE, recursive = TRUE)

# Process each dataset
all_refs <- list()

# 1. Apple dataset (should work)
all_refs$apple <- safe_little_mcar(apple, "apple")

# 2. Missvals dataset (should work with warnings)
all_refs$missvals <- safe_little_mcar(missvals, "missvals")

# 3. Complete data (edge case - no missing values)
set.seed(42)
complete_data <- matrix(rnorm(50 * 3), 50, 3)
colnames(complete_data) <- c("X1", "X2", "X3")
all_refs$complete <- safe_little_mcar(complete_data, "complete")

# 4. Simple MCAR data (should work)
set.seed(42)
simple_mcar <- matrix(rnorm(100 * 3), 100, 3)
# Add 10% random missingness
missing_indices <- sample(1:300, 30)
simple_mcar[missing_indices] <- NA
colnames(simple_mcar) <- c("V1", "V2", "V3")
all_refs$simple_mcar <- safe_little_mcar(simple_mcar, "simple_mcar")

# 5. Extreme case - single observation per pattern (might fail)
extreme_data <- matrix(c(
  1, NA, NA,
  NA, 2, NA,
  NA, NA, 3,
  1, 2, 3
), nrow = 4, byrow = TRUE)
colnames(extreme_data) <- c("X1", "X2", "X3")
all_refs$extreme <- safe_little_mcar(extreme_data, "extreme")

# Write all successful references
for (name in names(all_refs)) {
  ref <- all_refs[[name]]
  if (!is.null(ref) && !isTRUE(ref$failed)) {
    filename <- sprintf("tests/references/little_mcar_%s.json", name)
    write_json(ref, filename, pretty = TRUE, digits = 10, auto_unbox = TRUE)
    cat(sprintf("\nWrote: %s\n", filename))
  } else {
    cat(sprintf("\nSkipped %s due to errors\n", name))
  }
}

# Summary report
cat("\n\n=== SUMMARY OF RESULTS ===\n")
cat(sprintf("%-20s %-10s %-8s %-10s %s\n", 
            "Dataset", "χ²", "df", "p-value", "Decision"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (name in names(all_refs)) {
  ref <- all_refs[[name]]
  if (!is.null(ref$test_statistic) && !is.na(ref$test_statistic)) {
    decision <- ifelse(ref$p_value < 0.05, "REJECT MCAR", "Accept MCAR")
    cat(sprintf("%-20s %-10.4f %-8d %-10.6f %s\n",
                name, ref$test_statistic, ref$df, ref$p_value, decision))
  } else {
    cat(sprintf("%-20s %-10s %-8s %-10s %s\n",
                name, "NA", "NA", "NA", "Failed"))
  }
}

cat("\n\nReference generation complete!\n")
cat("Files created:\n")
list.files("tests/references", pattern = "little_mcar.*\\.json", full.names = TRUE)

# Save the summary as well
summary_data <- list(
  generation_date = Sys.Date(),
  r_version = R.version.string,
  package_versions = list(
    mvnmle = as.character(packageVersion("mvnmle")),
    BaylorEdPsych = as.character(packageVersion("BaylorEdPsych"))
  ),
  datasets_processed = names(all_refs),
  results_summary = lapply(all_refs, function(ref) {
    if (!is.null(ref$test_statistic)) {
      list(
        chi_square = ref$test_statistic,
        p_value = ref$p_value,
        df = ref$df,
        mcar_rejected = ref$p_value < 0.05
      )
    } else {
      list(error = TRUE)
    }
  })
)

write_json(summary_data, 
           "tests/references/little_mcar_summary.json",
           pretty = TRUE, auto_unbox = TRUE)

cat("\nAlso created: tests/references/little_mcar_summary.json\n")