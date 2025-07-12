# generate_references.R
library(mvnmle)
library(jsonlite)

# Load datasets
data(apple)
data(missvals)

# Generate ML estimates
cat("Computing R reference results...\n")
apple_result <- mlest(apple)
missvals_result <- mlest(missvals, iterlim = 400)

# Create additional test case
set.seed(42)
test_data <- matrix(rnorm(50), 10, 5)
test_data[sample(50, 10)] <- NA
small_result <- mlest(test_data)

# Format results for Python
format_result <- function(result) {
  list(
    muhat = as.vector(result$muhat),
    sigmahat = as.matrix(result$sigmahat),
    loglik = -result$value / 2,  # Convert from -2*loglik to loglik
    converged = result$code <= 2,
    iterations = result$iterations,
    gradient = as.vector(result$gradient),
    stop_code = result$code
  )
}

# Create reference data
references <- list(
  apple = format_result(apple_result),
  missvals = format_result(missvals_result),
  small_test = list(
    data = test_data,
    result = format_result(small_result)
  )
)

# Create output directory
dir.create("references", recursive = TRUE, showWarnings = FALSE)

# Save to JSON with high precision
write_json(references$apple, "references/apple_reference.json", 
           auto_unbox = TRUE, digits = 16, pretty = TRUE)
write_json(references$missvals, "references/missvals_reference.json", 
           auto_unbox = TRUE, digits = 16, pretty = TRUE)
write_json(references$small_test, "references/small_test_reference.json", 
           auto_unbox = TRUE, digits = 16, pretty = TRUE)

cat("Reference files generated successfully!\n")
cat("Files created:\n")
cat("  - tests/references/apple_reference.json\n")
cat("  - tests/references/missvals_reference.json\n")
cat("  - tests/references/small_test_reference.json\n")

# Print a sample to verify
cat("\nSample apple result:\n")
cat("Mean:", apple_result$muhat, "\n")
cat("Covariance diagonal:", diag(apple_result$sigmahat), "\n")
cat("Log-likelihood:", -apple_result$value/2, "\n")