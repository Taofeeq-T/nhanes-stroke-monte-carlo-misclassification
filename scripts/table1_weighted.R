################################################################################
# NHANES Stroke Prediction — Weighted Table 1 (Pre-Imputation)
################################################################################

# install.packages(c("tableone", "survey", "nhanesdata", "e1071"))

# --- Resolve the repository root (works from RStudio or Rscript) ---
.script_path <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) return(normalizePath(f[1]))
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (nzchar(p)) return(normalizePath(p))
  }
  ""
}
.repo_root <- function() {
  sp <- .script_path()
  if (nzchar(sp)) return(dirname(dirname(sp)))     # scripts/<file>.R -> repo root
  p <- normalizePath(getwd())
  while (!dir.exists(file.path(p, "scripts")) && dirname(p) != p) p <- dirname(p)
  p
}
setwd(.repo_root())
if (!dir.exists("scripts")) {
  stop("Could not find the repository root. Run setwd(\"/path/to/nhanes-stroke-monte-carlo-misclassification\") first.")
}
cat("Working directory:", getwd(), "\n")
dir.create("artifacts", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)

# --- Log console output to logs/stage2_table1.log (for review bundle) ---
dir.create("logs", showWarnings = FALSE)
.log_con <- file("logs/stage2_table1.log", open = "wt")
sink(.log_con, split = TRUE)


library(tableone)
library(survey)
library(nhanesdata)
library(dplyr)
library(e1071) # For skewness calculation

# ==============================================================================
# 1. LOAD DATA & CODEBOOK
# ==============================================================================

analytic <- read.csv("data/nhanes_stroke_analytic_2003_2023.csv", stringsAsFactors = TRUE)
codebook <- read.csv("artifacts/variable_codebook.csv", stringsAsFactors = FALSE)

cat("Loaded analytic dataset:", nrow(analytic), "rows\n")

# ==============================================================================
# 2. DYNAMIC VARIABLE CLASSIFICATION
# ==============================================================================

# 1. Categorical: All 'binary' or 'categorical' types (excluding outcome)
cat_vars <- codebook %>% 
  filter(type %in% c("binary", "categorical"), variable != "stroke") %>% 
  pull(variable)

# 2. Continuous: All 'continuous' types
cont_vars <- codebook %>% 
  filter(type == "continuous") %>% 
  pull(variable)

# 3. All variables for the table
all_vars <- codebook %>% 
  filter(variable != "stroke") %>% 
  pull(variable)

# 4. Automated Skewness Check (Identify non-normal variables)
# We calculate unweighted skewness as a proxy for distributional shape
skew_values <- sapply(analytic[, cont_vars], function(x) skewness(x, na.rm = TRUE))
nonnormal_vars <- names(skew_values[abs(skew_values) > 1])

cat("Variables identified as non-normal (Median [IQR]):\n", paste(nonnormal_vars, collapse=", "), "\n")

# ==============================================================================
# 3. DATA PREP & SURVEY DESIGN
# ==============================================================================

# Factorize categorical variables
for (v in cat_vars) {
  analytic[[v]] <- factor(analytic[[v]])
}

# Set race_eth reference level to NH White
if ("race_eth" %in% names(analytic)) {
  analytic$race_eth <- relevel(factor(analytic$race_eth), ref = "NH White")
}

# Label outcome for readability
analytic$stroke <- factor(analytic$stroke, levels = c(0, 1),
                          labels = c("No Stroke", "Stroke"))

# Build survey design: pooled weight (WTMEC2YR for 2003-2018, WTPH2YR for
# 2021-2023) divided by the number of cycles; PSUs nested within strata.
options(survey.lonely.psu = "adjust")
nhanes_design <- svydesign(ids = ~sdmvpsu, strata = ~sdmvstra, weights = ~wt_pooled,
                           nest = TRUE, data = analytic)

options(survey.lonely.psu = "adjust")

# ==============================================================================
# 4. CREATE WEIGHTED TABLE 1
# ==============================================================================

tab1 <- svyCreateTableOne(
  vars       = all_vars,
  strata     = "stroke",
  data       = nhanes_design,
  factorVars = cat_vars,
  addOverall = TRUE,
  test       = TRUE
)

# ==============================================================================
# 5. PRINT & EXPORT
# ==============================================================================

# Print to console for review
print(tab1, 
      nonnormal = nonnormal_vars, 
      smd = TRUE, 
      missing = TRUE, 
      formatOptions = list(big.mark = ","))

# Export for publication formatting
tab1_csv <- print(tab1,
                  nonnormal     = nonnormal_vars,
                  showAllLevels = TRUE,
                  missing       = TRUE,
                  smd           = TRUE,
                  printToggle   = FALSE)

write.csv(tab1_csv, "artifacts/table1_weighted_final.csv")
cat("\nSuccess: table1_weighted_final.csv saved.\n")

sink()
close(.log_con)