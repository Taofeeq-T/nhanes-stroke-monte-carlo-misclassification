################################################################################
# NHANES Stroke Prediction — Weighted Table 1 (Pre-Imputation)
################################################################################

# install.packages(c("tableone", "survey", "nhanesdata", "e1071"))

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

# Build survey design (MEC weights)
nhanes_design <- create_design(
  dsn      = analytic,
  start_yr = min(analytic$year),
  end_yr   = max(analytic$year),
  wt_type  = "mec"
)

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