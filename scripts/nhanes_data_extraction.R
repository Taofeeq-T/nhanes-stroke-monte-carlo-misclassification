################################################################################
# NHANES Stroke Prediction — Data Extraction & Assembly
# Pulls Tier 1 (GVRS-anchored) and Tier 2 predictors from NHANES 2003–2023
# Uses: nhanesdata package, survey weights via create_design()
# Author: Taofeeq O. Togunwa
################################################################################

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

# --- Log console output to logs/stage1_extraction.log (for review bundle) ---
dir.create("logs", showWarnings = FALSE)
.log_con <- file("logs/stage1_extraction.log", open = "wt")
sink(.log_con, split = TRUE)

# --- Remove stale imputation outputs from the previous pipeline ---
unlink("data/imputed_folds", recursive = TRUE)
unlink(c("data/nhanes_stroke_imputed_2003_2023.csv",
         "data/nhanes_stroke_imputed_2003_2023.parquet"))

# ==============================================================================
# 0. SETUP
# ==============================================================================

# install.packages("nhanesdata")           # CRAN (approved Feb 2026)
# install.packages(c("dplyr", "tidyr", "survey", "arrow", "naniar"))

library(nhanesdata)
library(dplyr)
library(tidyr)
library(naniar)   # missingness diagnostics

cat("nhanesdata version:", as.character(packageVersion("nhanesdata")), "\n")

# ==============================================================================
# 1. PULL RAW NHANES MODULES
# ==============================================================================
# nhanesdata::read_nhanes() returns all cycles merged with a `year` column
# (cycle start year: 2003, 2005, ..., 2017, 2021) and lowercase variable names.
# The 2019-2020 cycle is excluded automatically (COVID disruption).
#
# We pull each module once, then select only the variables we need.

cat("Downloading NHANES modules...\n")

# --- Demographics ---
demo <- read_nhanes("demo") |>
  select(
    seqn, year,
    ridageyr,          # Age in years at screening
    riagendr,          # Sex (1=Male, 2=Female)
    ridreth3,          # Race/ethnicity (6-level, available 2011+; see harmonization below)
    ridreth1,          # Race/ethnicity (5-level, available all cycles)
    dmdeduc2,          # Education level (adults 20+)
    indfmpir,          # Poverty-income ratio
    ridstatr,          # Interview/examination status (1 = interview only, 2 = interview + MEC)
    ridexprg,          # Pregnancy status at MEC exam (women 20-44)
    # Survey design variables
    sdmvpsu,           # Masked variance pseudo-PSU
    sdmvstra,          # Masked variance pseudo-stratum
    wtmec2yr,          # MEC exam weight (2-year)
    wtint2yr           # Interview weight (2-year)
  )

# --- Blood Pressure (Exam — auscultatory, 2003-2017/2020 pre-pandemic) ---
bpx <- read_nhanes("bpx") |>
  select(
    seqn, year,
    matches("^bpxsy[1-3]$"),    # Systolic BP readings 1-3 (auscultatory)
    matches("^bpxdi[1-3]$")     # Diastolic BP readings 1-3 (auscultatory)
  )

# --- Blood Pressure (Exam — oscillometric, 2017+ via BPXO table) ---
# Available from 2017-2018 onward; sole method from 2021-2023
bpxo <- tryCatch(
  read_nhanes("bpxo") |>
    select(
      seqn, year,
      matches("^bpxosy[1-3]$"),   # Systolic BP readings 1-3 (oscillometric)
      matches("^bpxodi[1-3]$")    # Diastolic BP readings 1-3 (oscillometric)
    ),
  error = function(e) {
    message("BPXO table not found; will rely on BPX auscultatory data.")
    NULL
  }
)

# --- Body Measures (Exam) ---
bmx <- read_nhanes("bmx") |>
  select(
    seqn, year,
    bmxbmi,            # Body mass index (kg/m^2)
    bmxwaist           # Waist circumference (cm)
  )

# --- Blood Pressure & Cholesterol Questionnaire ---
bpq <- read_nhanes("bpq") |>
  select(
    seqn, year,
    bpq020,            # Ever told you had high blood pressure?
    bpq050a,           # Currently taking prescribed medicine for HBP? (2003-2018)
    any_of("bpq150")   # Now taking prescribed medicine for HBP? (2021-2023)
  )

# --- Smoking (Questionnaire) ---
smq <- read_nhanes("smq") |>
  select(
    seqn, year,
    smq020,            # Smoked at least 100 cigarettes in life?
    smq040             # Do you now smoke cigarettes? (every day / some days / not at all)
  )

# --- Alcohol Use (Questionnaire) ---
# ALQ130 is drinks per day among those who drank in the past 12 months.
# Non-drinkers are routed away by skip pattern and show as NA in ALQ130.
# To distinguish non-drinkers (0 drinks) from true missingness, we also
# pull the screener questions ALQ101/ALQ111 (varied across cycles).
alq <- read_nhanes("alq") |>
  select(
    seqn, year,
    any_of(c(
      "alq130",          # Avg # drinks on days drinking, past 12 months
      "alq101",          # Had at least 12 drinks in any 1 year? (2003-2016)
      "alq110",          # Had at least 12 drinks in lifetime? (2003-2016)
      "alq111",          # Ever had a drink of any alcohol? (2017+)
      "alq120q",         # How often drank alcohol, past 12 months (2003-2016; 0 = none)
      "alq121"           # Past 12 months how often drank (2017+; "Never in the last year")
    ))
  )

# --- Physical Activity (Questionnaire) ---
# PAQ module changed substantially across NHANES:
#   2003-2006: PAD200 (vigorous), PAD320 (moderate) — minutes-based
#   2007-2017: PAQ605/620/650/665 — GPAQ yes/no format
#   2021-2023: PAD790Q/PAD800/PAD810Q/PAD820 — frequency/duration format
# We pull all available variables and harmonize to a binary indicator below.
paq <- read_nhanes("paq") |>
  select(
    seqn, year,
    any_of(c(
      # 2007-2017 format (Global PAQ)
      "paq605",          # Vigorous work activity? (Yes/No)
      "paq620",          # Moderate work activity?
      "paq650",          # Vigorous recreational activity?
      "paq665",          # Moderate recreational activity?
      # 2003-2006 format
      "pad200",          # Vigorous activity past 30 days (minutes)
      "pad320",          # Moderate activity past 30 days (minutes)
      # 2021-2023 format (redesigned LTPA)
      "pad790q",         # Frequency of moderate leisure-time PA
      "pad800",          # Minutes of moderate LTPA
      "pad810q",         # Frequency of vigorous leisure-time PA
      "pad820",          # Minutes of vigorous LTPA
      "pad790u",         # Unit for PAD790Q (day / week / month / year)
      "pad810u"          # Unit for PAD810Q
    ))
  )

# --- Diabetes (Questionnaire) ---
diq <- read_nhanes("diq") |>
  select(
    seqn, year,
    diq010             # Doctor told you have diabetes? (1=Yes, 2=No, 3=Borderline)
  )

# --- Medical Conditions (Questionnaire) — contains OUTCOME + comorbidities ---
mcq <- read_nhanes("mcq") |>
  select(
    seqn, year,
    mcq160f,           # Ever told you had a stroke? *** PRIMARY OUTCOME ***
    mcq160c            # Ever told you had coronary heart disease?
  )

# --- Sleep (Questionnaire) — available 2005+ ---
slq <- tryCatch(
  read_nhanes("slq") |>
    select(
      seqn, year,
      sld012             # Sleep hours on weekdays/workdays (continuous)
      # Note: In earlier cycles this may be SLD010H
    ),
  error = function(e) {
    message("SLQ table not available; sleep variable will be NA.")
    NULL
  }
)

# --- Glycohemoglobin (Lab) ---
# Standard table "ghb" covers 2005+. For 2003, LBXGH is in "l10_c".
# nhanesdata merges all cycles into "ghb" — check if 2003 is included.
# Helper: report per-cycle availability of a lab variable and patch a cycle from
# an alternative table if it is absent. Uses cat() so messages appear in the log.
.report_avail <- function(df, vars, label) {
  v <- intersect(vars, names(df))
  if (!length(v)) { cat("  [!]", label, ": none of", paste(vars, collapse = "/"), "present\n"); return(invisible()) }
  tab <- df |> filter(year %in% 2003:2021) |> group_by(year) |>
    summarise(non_missing = sum(!is.na(.data[[v[1]]])), .groups = "drop")
  cat("  ", label, " non-missing by cycle: ",
      paste(tab$year, tab$non_missing, sep = "=", collapse = ", "), "\n", sep = "")
}
.patch_cycle <- function(df, vars, alt_tables, cycle, label, cdc_tables = character()) {
  v <- intersect(vars, names(df))
  have <- length(v) > 0 && sum(!is.na(df[[v[1]]][df$year == cycle])) > 0
  if (have) return(df)
  for (tb in alt_tables) {
    alt <- tryCatch(read_nhanes(tb), error = function(e) {
      cat("  [!] could not read table '", tb, "': ", conditionMessage(e), "\n", sep = ""); NULL })
    if (is.null(alt)) next
    av <- intersect(vars, names(alt))
    if (!length(av)) { cat("  [!] '", tb, "' has none of ", paste(vars, collapse = "/"), "\n", sep = ""); next }
    alt <- alt |> filter(year == cycle) |> select(seqn, year, all_of(av))
    n_ok <- sum(!is.na(alt[[av[1]]]))
    if (n_ok > 0) {
      cat("  -> Patched ", label, " for ", cycle, " from '", tb, "' (", n_ok, " values)\n", sep = "")
      return(bind_rows(df |> filter(year != cycle), alt))
    }
    cat("  [!] '", tb, "' has no non-missing ", label, " for ", cycle, "\n", sep = "")
  }
  # Fall back to downloading the cycle-specific CDC table with nhanesA
  for (tb in cdc_tables) {
    alt <- tryCatch(nhanesA::nhanes(tb, translated = FALSE), error = function(e) {
      cat("  [!] nhanesA could not fetch '", tb, "': ", conditionMessage(e), "\n", sep = ""); NULL })
    if (is.null(alt)) next
    names(alt) <- tolower(names(alt))
    av <- intersect(vars, names(alt))
    if (!length(av)) {
      cat("  [!] '", tb, "' has none of ", paste(vars, collapse = "/"),
          "; available: ", paste(head(names(alt), 25), collapse = ", "), "\n", sep = ""); next
    }
    alt <- alt |> mutate(year = cycle) |> select(seqn, year, all_of(av))
    n_ok <- sum(!is.na(alt[[av[1]]]))
    if (n_ok > 0) {
      cat("  -> Patched ", label, " for ", cycle, " from CDC table '", tb, "' (", n_ok, " values)\n", sep = "")
      return(bind_rows(df |> filter(year != cycle), alt))
    }
  }
  cat("  [!] ", label, " remains missing for ", cycle, "\n", sep = "")
  df
}

ghb <- read_nhanes("ghb") |>
  select(seqn, year, any_of("lbxgh"))

.report_avail(ghb, "lbxgh", "HbA1c")
ghb <- .patch_cycle(ghb, "lbxgh", c("l10", "lab10"), 2003, "HbA1c", cdc_tables = "L10_C")

# --- Total Cholesterol (Lab) ---
# Standard table "tchol" covers 2005+. For 2003, LBXTC is in "l13_c".
tchol <- read_nhanes("tchol") |>
  select(seqn, year, any_of("lbxtc"))
.report_avail(tchol, "lbxtc", "Total cholesterol")
tchol <- .patch_cycle(tchol, "lbxtc", c("l13", "lab13"), 2003, "Total cholesterol",
                      cdc_tables = "L13_C")

# --- HDL Cholesterol (Lab) ---
# Standard table "hdl" covers 2005+. For 2003, HDL is also in "l13_c".
hdl <- read_nhanes("hdl") |>
  select(seqn, year, any_of(c("lbdhdd", "lbdhdl")))
.report_avail(hdl, c("lbdhdd", "lbdhdl"), "HDL cholesterol")
hdl <- .patch_cycle(hdl, c("lbdhdd", "lbdhdl", "lbxhdd"), c("l13", "lab13"), 2003,
                    "HDL cholesterol", cdc_tables = "L13_C")

# --- Triglycerides (Lab) ---
# Standard table "trigly" covers 2005+. For 2003, LBXTR is in "l13am_c".
# For 2021, variable was renamed from lbxtr to lbxtlg.
trigly <- read_nhanes("trigly") |>
  select(seqn, year, any_of(c("lbxtr", "lbxtlg")))

# Harmonize 2021 variable name: lbxtlg → lbxtr
if ("lbxtlg" %in% names(trigly)) {
  trigly <- trigly |>
    mutate(lbxtr = coalesce(
      if ("lbxtr" %in% names(trigly)) lbxtr else NA_real_,
      lbxtlg
    )) |>
    select(seqn, year, lbxtr)
} else {
  trigly <- trigly |> select(seqn, year, any_of("lbxtr"))
}

# Patch 2003 from l13am
if (!2003 %in% trigly$year[!is.na(trigly$lbxtr)]) {
  trigly_2003 <- tryCatch(
    read_nhanes("l13am") |>
      select(seqn, year, any_of("lbxtr")) |>
      filter(year == 2003),
    error = function(e) NULL
  )
  if (!is.null(trigly_2003) && nrow(trigly_2003) > 0) {
    trigly <- bind_rows(trigly |> filter(year != 2003), trigly_2003)
    cat("  → Patched Triglycerides for 2003 from l13am\n")
  }
}

# --- Standard Biochemistry (Lab) — contains creatinine ---
# Standard table "biopro" covers 2005+. For 2003, LBXSCR is in "l40_c".
biopro <- tryCatch(
  read_nhanes("biopro") |>
    select(seqn, year, any_of(c("lbxscr", "lbdscr"))),
  error = function(e) NULL
)

if (is.null(biopro)) {
  biopro <- data.frame(seqn = integer(), year = integer(), lbxscr = numeric())
}

scr_col <- intersect(c("lbxscr", "lbdscr"), names(biopro))[1]
if (!is.na(scr_col) && !2003 %in% biopro$year[!is.na(biopro[[scr_col]])]) {
  biopro_2003 <- tryCatch(
    read_nhanes("l40") |>
      select(seqn, year, any_of(c("lbxscr", "lbdscr"))) |>
      filter(year == 2003),
    error = function(e) NULL
  )
  if (!is.null(biopro_2003) && nrow(biopro_2003) > 0) {
    biopro <- bind_rows(biopro |> filter(year != 2003), biopro_2003)
    cat("  → Patched Creatinine for 2003 from l40\n")
  }
}

# --- Phlebotomy weight (2021-2023 only) ---
# For August 2021-August 2023, NCHS provides WTPH2YR, which must be used for
# analyses involving blood-based analytes. It is carried in the blood lab files.
wtph <- purrr::map(c("ghb", "cbc", "tchol", "hdl", "biopro"), function(tb) {
  tryCatch(read_nhanes(tb) |> filter(year == 2021) |> select(seqn, year, any_of("wtph2yr")),
           error = function(e) NULL)
}) |>
  purrr::compact() |>
  purrr::keep(~ "wtph2yr" %in% names(.x)) |>
  bind_rows() |>
  filter(!is.na(wtph2yr)) |>
  distinct(seqn, year, .keep_all = TRUE)
if (nrow(wtph) == 0) stop("WTPH2YR not found in 2021-2023 blood lab files; check nhanesdata tables.")
cat("  WTPH2YR found for", nrow(wtph), "participants in 2021-2023\n")

# --- Complete Blood Count (Lab) ---
# Standard table "cbc" covers 2005+. For 2003, LBXWBCSI is in "l25_c".
cbc <- read_nhanes("cbc") |>
  select(seqn, year, any_of("lbxwbcsi"))

if (!2003 %in% cbc$year[!is.na(cbc$lbxwbcsi)]) {
  cbc_2003 <- tryCatch(
    read_nhanes("l25") |>
      select(seqn, year, any_of("lbxwbcsi")) |>
      filter(year == 2003),
    error = function(e) NULL
  )
  if (!is.null(cbc_2003) && nrow(cbc_2003) > 0) {
    cbc <- bind_rows(cbc |> filter(year != 2003), cbc_2003)
    cat("  → Patched WBC for 2003 from l25\n")
  }
}

# --- Fasting Glucose: EXCLUDED ---
# Fasting glucose is structurally missing by protocol (~50% of MEC participants
# are in the fasting subsample). This is not MAR/MCAR — it is missing by design.
# HbA1c (non-fasting, available for all MEC participants) fully covers the
# glycemic domain and is the AHA Life's Essential 8 preferred measure.
# Imputation of protocol-driven missingness is methodologically inappropriate.

cat("All modules downloaded.\n")

# ==============================================================================
# 2. FILTER TO 2003+ AND MERGE
# ==============================================================================

# Filter all modules to 2003+ (cycle start year >= 2003)
filter_year <- function(df) {
  if (is.null(df)) return(NULL)
  df |> filter(year >= 2003)
}

demo  <- filter_year(demo)
bpx   <- filter_year(bpx)
bpxo  <- filter_year(bpxo)
bmx   <- filter_year(bmx)
bpq   <- filter_year(bpq)
smq   <- filter_year(smq)
alq   <- filter_year(alq)
paq   <- filter_year(paq)
diq   <- filter_year(diq)
mcq   <- filter_year(mcq)
slq   <- filter_year(slq)
ghb   <- filter_year(ghb)
tchol <- filter_year(tchol)
hdl   <- filter_year(hdl)
trigly <- filter_year(trigly)
biopro <- filter_year(biopro)
cbc   <- filter_year(cbc)

# Sequential left joins on (seqn, year)
# Start with demo (defines the cohort universe)
join_keys <- c("seqn", "year")

dat <- demo |>
  left_join(bpx,    by = join_keys) |>
  left_join(bmx,    by = join_keys) |>
  left_join(bpq,    by = join_keys) |>
  left_join(smq,    by = join_keys) |>
  left_join(alq,    by = join_keys) |>
  left_join(paq,    by = join_keys) |>
  left_join(wtph,   by = join_keys) |>
  left_join(diq,    by = join_keys) |>
  left_join(mcq,    by = join_keys) |>
  left_join(ghb,    by = join_keys) |>
  left_join(tchol,  by = join_keys) |>
  left_join(hdl,    by = join_keys) |>
  left_join(trigly, by = join_keys) |>
  left_join(cbc,    by = join_keys)

# Conditionally join oscillometric BP, sleep, biochemistry
if (!is.null(bpxo))   dat <- dat |> left_join(bpxo,   by = join_keys)
if (!is.null(slq))    dat <- dat |> left_join(slq,    by = join_keys)
if (!is.null(biopro)) dat <- dat |> left_join(biopro, by = join_keys)

cat("Merged dataset:", nrow(dat), "rows x", ncol(dat), "columns\n")

# ==============================================================================
# 3. RESTRICT TO ADULTS 20+ WITH NON-MISSING OUTCOME
# ==============================================================================

# First restrict to adults 20+
dat <- dat |> filter(ridageyr >= 20)

# Tag participants by outcome response status BEFORE dropping exclusions
# NOTE: nhanesdata returns pre-translated text labels, not numeric codes
dat <- dat |>
  mutate(
    outcome_status = case_when(
      mcq160f == "Yes"        ~ "Stroke (Yes)",
      mcq160f == "No"         ~ "No stroke (No)",
      mcq160f == "Refused"    ~ "Refused",
      mcq160f == "Don't know" ~ "Don't know",
      is.na(mcq160f)          ~ "Missing/Not asked",
      TRUE                    ~ "Other"
    )
  )

cat("Adults 20+ by MCQ160F response status:\n")
print(table(dat$outcome_status, useNA = "always"))

# --- Characterize excluded vs. included (for Supplementary Table) ---
# Compare demographics of those excluded (refused/don't know/missing)
# against those with a valid yes/no response. Systematic differences
# suggest potential selection bias.

dat <- dat |>
  mutate(
    included = mcq160f %in% c("Yes", "No"),
    # Pre-derive a few comparison variables before full harmonization
    .age     = ridageyr,
    .sex     = as.character(riagendr),   # Already "Male"/"Female" from nhanesdata
    .race    = as.character(ridreth1),   # Already text labels
    .educ    = case_when(
      grepl("Less [Tt]han 9th", dmdeduc2)         ~ "Less than HS",
      grepl("9-11th [Gg]rade", dmdeduc2)          ~ "Less than HS",
      grepl("[Hh]igh [Ss]chool", dmdeduc2)        ~ "HS/GED",
      grepl("[Ss]ome [Cc]ollege", dmdeduc2)       ~ "Some college",
      grepl("[Cc]ollege [Gg]raduate", dmdeduc2)   ~ "College+",
      TRUE                                         ~ NA_character_
    ),
    .pir     = indfmpir
  )

cat("\n========================================\n")
cat("EXCLUSION CHARACTERIZATION\n")
cat("(Supplementary Table: Included vs. Excluded)\n")
cat("========================================\n")

n_included <- sum(dat$included)
n_excluded <- sum(!dat$included)
cat(sprintf("\nIncluded (valid Yes/No): %d\n", n_included))
cat(sprintf("Excluded (Refused/DK/Missing): %d (%.1f%%)\n",
            n_excluded, n_excluded / nrow(dat) * 100))

# Minimum non-missing observations needed for statistical tests
min_n_for_test <- 5

# Age comparison
cat("\nAge (mean ± SD):\n")
age_inc <- dat$.age[dat$included]
age_exc <- dat$.age[!dat$included]
cat(sprintf("  Included: %.1f ± %.1f (n=%d)\n",
            mean(age_inc, na.rm = TRUE), sd(age_inc, na.rm = TRUE), sum(!is.na(age_inc))))
cat(sprintf("  Excluded: %.1f ± %.1f (n=%d)\n",
            mean(age_exc, na.rm = TRUE), sd(age_exc, na.rm = TRUE), sum(!is.na(age_exc))))
if (sum(!is.na(age_exc)) >= min_n_for_test & sum(!is.na(age_inc)) >= min_n_for_test) {
  wt <- wilcox.test(age_inc, age_exc)
  cat(sprintf("  Wilcoxon p = %.4f\n", wt$p.value))
} else {
  cat("  (Too few excluded observations for statistical test)\n")
}

# Sex comparison
cat("\nSex (% Female):\n")
sex_tbl <- table(dat$.sex, dat$included)
if (ncol(sex_tbl) == 2 & nrow(sex_tbl) == 2 & sum(sex_tbl[, FALSE]) >= min_n_for_test) {
  pct_f_inc <- sex_tbl["Female", TRUE] / sum(sex_tbl[, TRUE]) * 100
  pct_f_exc <- sex_tbl["Female", FALSE] / sum(sex_tbl[, FALSE]) * 100
  cat(sprintf("  Included: %.1f%%\n", pct_f_inc))
  cat(sprintf("  Excluded: %.1f%%\n", pct_f_exc))
  cat(sprintf("  Chi-sq p = %.4f\n", chisq.test(sex_tbl)$p.value))
} else {
  cat("  (Too few excluded observations for comparison)\n")
}

# Race/ethnicity comparison
cat("\nRace/ethnicity distribution:\n")
race_tbl <- table(dat$.race, dat$included)
if (ncol(race_tbl) == 2 & sum(race_tbl[, FALSE]) >= min_n_for_test) {
  cat("  Included:\n")
  inc_pcts <- prop.table(race_tbl[, TRUE]) * 100
  for (r in names(inc_pcts)) cat(sprintf("    %-20s: %5.1f%%\n", r, inc_pcts[r]))
  cat("  Excluded:\n")
  exc_pcts <- prop.table(race_tbl[, FALSE]) * 100
  for (r in names(exc_pcts)) cat(sprintf("    %-20s: %5.1f%%\n", r, exc_pcts[r]))
  # Use Fisher's exact test for small cell counts, chi-sq otherwise
  if (any(race_tbl < 5)) {
    cat(sprintf("  Fisher p = %.4f\n", fisher.test(race_tbl, simulate.p.value = TRUE)$p.value))
  } else {
    cat(sprintf("  Chi-sq p = %.4f\n", chisq.test(race_tbl)$p.value))
  }
} else {
  cat("  (Too few excluded observations for comparison)\n")
}

# Education comparison
cat("\nEducation distribution:\n")
educ_tbl <- table(dat$.educ, dat$included, useNA = "no")
if (ncol(educ_tbl) == 2 & nrow(educ_tbl) > 1 & sum(educ_tbl[, FALSE]) >= min_n_for_test) {
  cat("  Included:\n")
  inc_e <- prop.table(educ_tbl[, TRUE]) * 100
  for (e in names(inc_e)) cat(sprintf("    %-20s: %5.1f%%\n", e, inc_e[e]))
  cat("  Excluded:\n")
  exc_e <- prop.table(educ_tbl[, FALSE]) * 100
  for (e in names(exc_e)) cat(sprintf("    %-20s: %5.1f%%\n", e, exc_e[e]))
  if (any(educ_tbl < 5)) {
    cat(sprintf("  Fisher p = %.4f\n", fisher.test(educ_tbl, simulate.p.value = TRUE)$p.value))
  } else {
    cat(sprintf("  Chi-sq p = %.4f\n", chisq.test(educ_tbl)$p.value))
  }
} else {
  cat("  (Too few excluded observations for comparison)\n")
}

# Poverty-income ratio comparison
cat("\nPoverty-income ratio (mean ± SD):\n")
pir_inc <- dat$.pir[dat$included]
pir_exc <- dat$.pir[!dat$included]
cat(sprintf("  Included: %.2f ± %.2f (n=%d)\n",
            mean(pir_inc, na.rm = TRUE), sd(pir_inc, na.rm = TRUE), sum(!is.na(pir_inc))))
cat(sprintf("  Excluded: %.2f ± %.2f (n=%d)\n",
            mean(pir_exc, na.rm = TRUE), sd(pir_exc, na.rm = TRUE), sum(!is.na(pir_exc))))
if (sum(!is.na(pir_exc)) >= min_n_for_test & sum(!is.na(pir_inc)) >= min_n_for_test) {
  wt_pir <- wilcox.test(pir_inc, pir_exc)
  cat(sprintf("  Wilcoxon p = %.4f\n", wt_pir$p.value))
} else {
  cat("  (Too few excluded observations for statistical test)\n")
}

# Save exclusion comparison as CSV for supplementary table
exclusion_summary <- dat |>
  group_by(included) |>
  summarise(
    n             = n(),
    age_mean      = round(mean(.age, na.rm = TRUE), 1),
    age_sd        = round(sd(.age, na.rm = TRUE), 1),
    pct_female    = round(mean(.sex == "Female", na.rm = TRUE) * 100, 1),
    pct_nh_white  = round(mean(grepl("Non-Hispanic White", .race), na.rm = TRUE) * 100, 1),
    pct_nh_black  = round(mean(grepl("Non-Hispanic Black", .race), na.rm = TRUE) * 100, 1),
    pct_college   = round(mean(.educ == "College+", na.rm = TRUE) * 100, 1),
    pir_mean      = round(mean(.pir, na.rm = TRUE), 2),
    pir_sd        = round(sd(.pir, na.rm = TRUE), 2),
    .groups = "drop"
  )

write.csv(exclusion_summary, "artifacts/exclusion_characterization.csv", row.names = FALSE)
cat("\nExported: exclusion_characterization.csv\n")

# --- Now apply the exclusion ---
dat <- dat |>
  filter(included) |>
  select(-included, -outcome_status, -.age, -.sex, -.race, -.educ, -.pir)

cat("\nAfter restricting to valid stroke response (Yes/No):",
    nrow(dat), "participants\n")

# ------------------------------------------------------------------------------
# 3b. RESTRICT TO PARTICIPANTS ELIGIBLE FOR THE MEC AND LABORATORY COMPONENTS
# ------------------------------------------------------------------------------
# The model uses examination (BP, body measures) and blood laboratory predictors,
# so the target population is adults who completed the MEC examination.
#   - RIDSTATR = 2 (interviewed and MEC-examined), equivalently WTMEC2YR > 0
#   - Not pregnant at the MEC examination (RIDEXPRG)
#   - 2021-2023: WTPH2YR > 0 (phlebotomy weight required for blood analytes)

is_mec   <- function(x) grepl("MEC", as.character(x), ignore.case = TRUE) | as.character(x) == "2"
is_preg  <- function(x) grepl("^Yes", as.character(x)) | as.character(x) == "1"

# RIDSTATR by cycle among adults with a valid outcome (reported in Supplement)
ridstatr_by_cycle <- dat |>
  count(year, ridstatr) |>
  group_by(year) |> mutate(pct = round(100 * n / sum(n), 1)) |> ungroup()
print(ridstatr_by_cycle)
write.csv(ridstatr_by_cycle, "artifacts/ridstatr_by_cycle.csv", row.names = FALSE)

flow <- list(tibble::tibble(step = "Adults 20+ with valid stroke response", year = dat$year))
dat <- dat |> filter(is_mec(ridstatr), !is.na(wtmec2yr), wtmec2yr > 0)
flow <- c(flow, list(tibble::tibble(step = "MEC-examined (RIDSTATR = 2, WTMEC2YR > 0)", year = dat$year)))
dat <- dat |> filter(!(is_preg(ridexprg) %in% TRUE))
flow <- c(flow, list(tibble::tibble(step = "Not pregnant at MEC examination", year = dat$year)))
dat <- dat |> filter(year < 2021 | (!is.na(wtph2yr) & wtph2yr > 0))
flow <- c(flow, list(tibble::tibble(step = "2021-2023: phlebotomy weight > 0", year = dat$year)))

cohort_flow <- bind_rows(flow) |>
  count(step, year) |>
  tidyr::pivot_wider(names_from = year, values_from = n) |>
  mutate(total = rowSums(across(-step), na.rm = TRUE))
cohort_flow <- cohort_flow[match(unique(bind_rows(flow)$step), cohort_flow$step), ]
print(cohort_flow)
write.csv(cohort_flow, "artifacts/cohort_flow_by_cycle.csv", row.names = FALSE)
cat("Final analytic cohort:", nrow(dat), "participants;",
    sum(dat$mcq160f == "Yes"), "stroke cases\n")
cat("Stroke cases (Yes):", sum(dat$mcq160f == "Yes"), "\n")
cat("Non-stroke (No):",   sum(dat$mcq160f == "No"), "\n")

# ==============================================================================
# 4. HARMONIZE VARIABLES ACROSS CYCLES
# ==============================================================================

if (!"bpq150" %in% names(dat)) dat$bpq150 <- NA_character_
dat$.bpq150 <- as.character(dat$bpq150)

dat <- dat |>
  mutate(
    # ---- OUTCOME ----
    # nhanesdata returns "Yes"/"No" text labels
    stroke = as.integer(mcq160f == "Yes"),  # 1 = Yes, 0 = No
    
    # ---- DEMOGRAPHICS ----
    age       = pmin(ridageyr, 80),   # 2003-2006 top-coded at 85; harmonize to 80
    sex       = factor(riagendr, levels = c("Male", "Female")),
    
    # Race/ethnicity: use ridreth1 (5-level, available all cycles)
    # nhanesdata returns full text labels; recode to shorter names
    race_eth  = case_when(
      ridreth1 == "Mexican American"                  ~ "Mexican American",
      ridreth1 == "Other Hispanic"                    ~ "Other Hispanic",
      ridreth1 == "Non-Hispanic White"                ~ "NH White",
      ridreth1 == "Non-Hispanic Black"                ~ "NH Black",
      ridreth1 == "Other Race - Including Multi-Racial" ~ "Other/Multi",
      TRUE                                            ~ NA_character_
    ),
    race_eth  = factor(race_eth, levels = c(
      "NH White", "NH Black", "Mexican American", "Other Hispanic", "Other/Multi"
    )),  # NH White as reference level (standard epidemiologic convention)
    
    # Education: nhanesdata labels vary in capitalization across cycles
    # Use grepl() for robust matching
    education = case_when(
      grepl("Less [Tt]han 9th", dmdeduc2)         ~ "Less than HS",
      grepl("9-11th [Gg]rade", dmdeduc2)          ~ "Less than HS",
      grepl("[Hh]igh [Ss]chool", dmdeduc2)        ~ "HS/GED",
      grepl("[Ss]ome [Cc]ollege", dmdeduc2)       ~ "Some college",
      grepl("[Cc]ollege [Gg]raduate", dmdeduc2)   ~ "College+",
      TRUE                                         ~ NA_character_
    ),
    education = factor(education, levels = c(
      "Less than HS", "HS/GED", "Some college", "College+"
    )),
    
    pir = indfmpir,   # Poverty-income ratio (continuous, 0-5)
  ) |>
  # ---- BLOOD PRESSURE: Convert 0 mmHg to NA ----
# NHANES codes DBP as 0 when Korotkoff sounds are heard to 0 mmHg
# (very rare) or for measurement failures. Including 0 in averages
# would produce biologically implausible values.
mutate(across(matches("^bpx(o)?(sy|di)[1-3]$"), ~na_if(., 0))) |>
  mutate(
    # ---- BLOOD PRESSURE (harmonize auscultatory + oscillometric) ----
    # Strategy: For 2003-2015, use auscultatory (BPXSY/DI).
    # For 2017+ pre-pandemic, use oscillometric (BPXOSY/ODI) as primary.
    # For 2017-2018 specifically, both are available; prefer oscillometric
    # for forward-compatibility with 2021-2023.
    # Average of readings 2 and 3 (dropping reading 1 per NHANES protocol).
    
    # Auscultatory average (readings 2 & 3, excluding reading 1)
    sbp_ausc = rowMeans(
      across(any_of(c("bpxsy2", "bpxsy3")), .fns = identity),
      na.rm = TRUE
    ),
    dbp_ausc = rowMeans(
      across(any_of(c("bpxdi2", "bpxdi3")), .fns = identity),
      na.rm = TRUE
    ),
    
    # Oscillometric average (readings 2 & 3)
    sbp_osci = rowMeans(
      across(any_of(c("bpxosy2", "bpxosy3")), .fns = identity),
      na.rm = TRUE
    ),
    dbp_osci = rowMeans(
      across(any_of(c("bpxodi2", "bpxodi3")), .fns = identity),
      na.rm = TRUE
    )
  ) |>
  mutate(
    # Replace NaN from rowMeans(all NA) with NA
    sbp_ausc = ifelse(is.nan(sbp_ausc), NA_real_, sbp_ausc),
    dbp_ausc = ifelse(is.nan(dbp_ausc), NA_real_, dbp_ausc),
    sbp_osci = ifelse(is.nan(sbp_osci), NA_real_, sbp_osci),
    dbp_osci = ifelse(is.nan(dbp_osci), NA_real_, dbp_osci),
    
    # Harmonized BP: prefer oscillometric when available (2017+), else auscultatory
    # NOTE: For publication, document this and consider calibration adjustment.
    # The 2017-2018 methodology study showed prevalence estimates were comparable
    # between protocols, so a simple preference rule is defensible.
    mean_sbp = case_when(
      year >= 2017 & !is.na(sbp_osci) ~ sbp_osci,
      TRUE                             ~ sbp_ausc
    ),
    mean_dbp = case_when(
      year >= 2017 & !is.na(dbp_osci) ~ dbp_osci,
      TRUE                             ~ dbp_ausc
    ),
    
    # ---- ANTIHYPERTENSIVE MEDICATION ----
    # BPQ020: Ever told HBP? BPQ050A: Currently taking meds?
    # nhanesdata returns "Yes"/"No" text labels
    htn_med = case_when(
      bpq050a == "Yes" ~ 1L,   # 2003-2018: taking meds
      .bpq150 == "Yes" ~ 1L,   # 2021-2023: BPQ050A replaced by BPQ150
      bpq020 == "No"   ~ 0L,   # Never told HBP -> not on meds
      bpq050a == "No"  ~ 0L,   # Told HBP but not on meds
      .bpq150 == "No"  ~ 0L,
      TRUE             ~ NA_integer_
    ),
    
    # ---- SMOKING STATUS ----
    # SMQ020: Smoked 100+ cigs in life? SMQ040: Now smoke?
    # SMQ040 labels vary across cycles: "Every day" vs "Every day,"
    # Use grepl() for robust matching
    smoking = case_when(
      smq020 == "No"                              ~ "Never",
      smq020 == "Yes" & grepl("Not at all", smq040) ~ "Former",
      smq020 == "Yes" & (grepl("Every day", smq040) |
                           grepl("Some days", smq040)) ~ "Current",
      TRUE                                        ~ NA_character_
    ),
    smoking = factor(smoking, levels = c("Never", "Former", "Current")),
    
    # ---- ALCOHOL ----
    # ALQ130 = average number of drinks on the days the participant drank, past
    # 12 months (drinks per DRINKING day, not per calendar day). It is asked only
    # of past-year drinkers. Cycle-specific rules (in order of precedence):
    #   1. Valid ALQ130 (< 100; 777/999 = Refused/Don't know -> missing)
    #   2. No drinking in the past 12 months -> 0
    #        2003-2016: ALQ120Q = 0;  2017+: ALQ121 = "Never in the last year"
    #   3. Lifetime non-drinker -> 0
    #        2003-2016: ALQ110 = "No" (<12 drinks in lifetime)
    #        2017+:     ALQ111 = "No" (never had a drink)
    #   4. 2003-2016: ALQ101 = "No" with no ALQ120Q/ALQ130 response -> 0
    #   5. Otherwise missing
    alcohol_drinks_per_day = {
      a130  <- suppressWarnings(as.numeric(as.character(alq130)))
      a120q <- suppressWarnings(as.numeric(as.character(alq120q)))
      a121  <- as.character(alq121)
      case_when(
        !is.na(a130) & a130 < 100                                   ~ a130,
        !is.na(a120q) & a120q == 0                                  ~ 0,
        a121 %in% c("Never in the last year", "0")                  ~ 0,
        as.character(alq110) %in% c("No", "2")                      ~ 0,
        as.character(alq111) %in% c("No", "2")                      ~ 0,
        as.character(alq101) %in% c("No", "2") & is.na(a120q) & is.na(a130) ~ 0,
        TRUE                                                        ~ NA_real_
      )
    },
    
    # ---- PHYSICAL ACTIVITY (harmonized binary indicator) ----
    # Harmonized to ANY MODERATE OR VIGOROUS LEISURE-TIME ACTIVITY in all cycles:
    #   2003-2006: PAD200 / PAD320 (Yes/No/Unable; leisure time or school,
    #              past 30 days, >=10 min). Occupational and transport items
    #              (PAD020, PAQ100) are NOT used.
    #   2007-2017: GPAQ recreational items PAQ650 / PAQ665 (Yes/No, typical week).
    #              Work items PAQ605 / PAQ620 are NOT used.
    #   2021-2023: PAD790Q / PAD810Q (frequency, with units PAD790U / PAD810U)
    #              and PAD800 / PAD820 (minutes) for moderate / vigorous
    #              leisure-time activity. Frequency is converted to sessions per
    #              week and, to match the GPAQ "typical week" threshold used in
    #              2007-2018, at least one session per week of at least 10
    #              minutes is required.
    # Refused / Don't know (7, 9, 77, 99, 7777, 9999) are treated as missing.
    phys_active = {
      yn <- function(x) {                       # "Yes"->1, "No"/"Unable"->0, else NA
        x <- as.character(x)
        dplyr::case_when(x %in% c("Yes", "1") ~ 1L,
                         x %in% c("No", "2", "Unable to do activity", "3") ~ 0L,
                         TRUE ~ NA_integer_)
      }
      num <- function(x) {                      # numeric, with refusal codes -> NA
        x <- suppressWarnings(as.numeric(as.character(x)))
        dplyr::if_else(x %in% c(7777, 9999, 77777, 99999), NA_real_, x)
      }
      rec_0716 <- pmax(yn(paq650), yn(paq665), na.rm = TRUE)
      lt_0306  <- pmax(yn(pad200), yn(pad320), na.rm = TRUE)
      per_week <- function(q, u) {                    # sessions per week
        u <- tolower(as.character(u))
        dplyr::case_when(
          grepl("^d", u) ~ num(q) * 7,
          grepl("^w", u) ~ num(q),
          grepl("^m", u) ~ num(q) / 4.345,
          grepl("^y", u) ~ num(q) / 52.143,
          TRUE           ~ num(q)                     # unit missing: assume weekly
        )
      }
      active_21 <- function(q, u, mins) {
        f <- per_week(q, u); m <- num(mins)
        dplyr::if_else(!is.na(f) & f >= 1 & !is.na(m) & m >= 10, 1L,
                       dplyr::if_else(!is.na(f), 0L, NA_integer_))
      }
      mod_21   <- active_21(pad790q, pad790u, pad800)
      vig_21   <- active_21(pad810q, pad810u, pad820)
      lt_2123  <- pmax(mod_21, vig_21, na.rm = TRUE)
      dplyr::case_when(
        year %in% c(2003, 2005)        ~ lt_0306,
        year >= 2007 & year <= 2017    ~ rec_0716,
        year >= 2021                   ~ lt_2123,
        TRUE                           ~ NA_integer_
      )
    },
    
    # ---- SELF-REPORTED DIABETES ----
    diabetes_sr = case_when(
      diq010 == "Yes"        ~ 1L,
      diq010 == "No"         ~ 0L,
      diq010 == "Borderline" ~ 0L,   # Group with No for binary
      TRUE                   ~ NA_integer_
    ),
    
    # ---- SELF-REPORTED CHD ----
    chd_sr = case_when(
      mcq160c == "Yes" ~ 1L,
      mcq160c == "No"  ~ 0L,
      TRUE             ~ NA_integer_
    ),
    
    # ---- HDL (harmonize variable name) ----
    # Variable name changed across cycles: lbdhdd vs lbdhdl
    # Use coalesce with across() + any_of() for safe column selection
    hdl_c = do.call(coalesce, across(any_of(c("lbdhdd", "lbdhdl", "lbxhdd")))),
    
    # ---- SERUM CREATININE (harmonize variable name) ----
    creatinine = do.call(coalesce, across(any_of(c("lbxscr", "lbdscr")))),
    
    # ---- SLEEP ----
    sleep_hrs = case_when(
      !is.null(slq) & !is.na(sld012) & sld012 <= 24 ~ as.numeric(sld012),
      TRUE ~ NA_real_
    ),
    
    # ---- RENAME LABS FOR CLARITY ----
    hba1c          = lbxgh,
    total_chol     = lbxtc,
    triglycerides  = lbxtr,
    wbc            = lbxwbcsi
  )

# ==============================================================================
# 4b. BLOOD PRESSURE METHOD COMPARISON (2017-2018 paired measurements)
# ==============================================================================
# In 2017-2018 both auscultatory (BPX) and oscillometric (BPXO) readings exist.
# Participants with both provide a within-person comparison of the two methods.
# Descriptive only (no calibration is applied to mean_sbp / mean_dbp).

bp_pair <- dat |>
  filter(year == 2017, age >= 20, !is.na(sbp_ausc), !is.na(sbp_osci),
         !is.na(dbp_ausc), !is.na(dbp_osci)) |>
  transmute(d_sbp = sbp_osci - sbp_ausc, d_dbp = dbp_osci - dbp_ausc)

bp_method_comparison <- tibble::tibble(
  measure    = c("SBP", "DBP"),
  n_paired   = nrow(bp_pair),
  mean_diff  = c(mean(bp_pair$d_sbp), mean(bp_pair$d_dbp)),   # oscillometric - auscultatory
  sd_diff    = c(sd(bp_pair$d_sbp),   sd(bp_pair$d_dbp)),
  loa_lower  = mean_diff - 1.96 * sd_diff,                     # Bland-Altman limits
  loa_upper  = mean_diff + 1.96 * sd_diff
)
print(bp_method_comparison)
write.csv(bp_method_comparison, "artifacts/bp_method_comparison_2017.csv", row.names = FALSE)

# ==============================================================================
# 5. SELECT ANALYTIC VARIABLES
# ==============================================================================

analytic <- dat |>
  select(
    # Identifiers & design
    seqn, year,
    sdmvpsu, sdmvstra, wtmec2yr, wtint2yr, wtph2yr,
    
    # Outcome
    stroke,
    
    # Tier 1: GVRS core components
    age,
    sex,
    race_eth,
    mean_sbp,
    mean_dbp,
    htn_med,
    hba1c,
    total_chol,
    hdl_c,
    smoking,
    bmxwaist,
    bmxbmi,
    alcohol_drinks_per_day,
    phys_active,
    
    # Tier 2: Established vascular risk markers beyond GVRS
    triglycerides,
    creatinine,
    wbc,
    education,
    pir,
    sleep_hrs,
    diabetes_sr,
    chd_sr
  )

cat("\nAnalytic dataset:", nrow(analytic), "rows x", ncol(analytic), "columns\n")

# ==============================================================================
# 6. MISSINGNESS PROFILING
# ==============================================================================
# This is Step 1 of the tiered missingness decision framework.
# Output: per-variable and per-cycle missingness rates.

cat("\n========================================\n")
cat("MISSINGNESS PROFILE\n")
cat("========================================\n")

# --- Overall missingness by variable ---
predictor_vars <- c(
  "age", "sex", "race_eth", "mean_sbp", "mean_dbp", "htn_med",
  "hba1c", "total_chol", "hdl_c", "smoking", "bmxwaist", "bmxbmi",
  "alcohol_drinks_per_day", "phys_active",
  "triglycerides", "creatinine", "wbc", "education", "pir",
  "sleep_hrs", "diabetes_sr", "chd_sr"
)

overall_miss <- analytic |>
  summarise(across(
    all_of(predictor_vars),
    list(
      n_miss  = ~sum(is.na(.)),
      pct_miss = ~round(mean(is.na(.)) * 100, 1)
    )
  )) |>
  pivot_longer(
    everything(),
    names_to = c("variable", ".value"),
    names_pattern = "(.+)_(n_miss|pct_miss)"
  ) |>
  arrange(desc(pct_miss))

cat("\nOverall missingness (sorted by % missing):\n")
print(as.data.frame(overall_miss), row.names = FALSE)

# --- Missingness by cycle (identifies structural vs sporadic) ---
cycle_miss <- analytic |>
  group_by(year) |>
  summarise(
    n = n(),
    across(
      all_of(predictor_vars),
      ~round(mean(is.na(.)) * 100, 1),
      .names = "{.col}"
    )
  ) |>
  arrange(year)

cat("\nMissingness (%) by NHANES cycle:\n")
print(as.data.frame(cycle_miss))

# --- Flag structural missingness ---
cat("\n--- Structural missingness flags ---\n")
cat("Variables with >90% missing in any cycle indicate structural absence:\n")

for (v in predictor_vars) {
  cycle_pcts <- cycle_miss[[v]]
  cycle_yrs  <- cycle_miss$year
  if (any(cycle_pcts > 90, na.rm = TRUE)) {
    bad <- cycle_yrs[which(cycle_pcts > 90)]
    cat(sprintf("  %-25s: >90%% missing in cycles %s\n",
                v, paste(bad, collapse = ", ")))
  }
}

# --- Missingness association with outcome ---
cat("\n--- Missingness vs. stroke outcome (chi-sq p-values) ---\n")
cat("Tests whether missingness pattern differs by stroke status:\n")

for (v in predictor_vars) {
  miss_indicator <- is.na(analytic[[v]])
  if (sum(miss_indicator) > 0 & sum(miss_indicator) < nrow(analytic)) {
    tbl <- table(miss_indicator, analytic$stroke)
    if (all(dim(tbl) == c(2, 2))) {
      p <- chisq.test(tbl)$p.value
      flag <- ifelse(p < 0.05, " ***", "")
      cat(sprintf("  %-25s: p = %.4f%s\n", v, p, flag))
    }
  }
}
cat("  (*** = p < 0.05; indicates missingness is associated with the outcome.\n")
cat("   This test cannot distinguish MAR from MNAR, and p >= 0.05 does not establish MCAR.)\n")

# ==============================================================================
# 7. TIERED MISSINGNESS DECISION
# ==============================================================================
# Apply the framework from our discussion:
#   <5%:   Impute without concern
#   5-30%: Impute, verify diagnostics
#   30-50%: Impute, run sensitivity ± variable
#   >50%: Exclude from primary model

cat("\n========================================\n")
cat("TIERED MISSINGNESS DECISION\n")
cat("========================================\n")

overall_miss <- overall_miss |>
  mutate(
    tier = case_when(
      pct_miss < 5   ~ "Tier A: Impute freely (<5%)",
      pct_miss < 30  ~ "Tier B: Impute, verify diagnostics (5-30%)",
      pct_miss < 50  ~ "Tier C: Impute + sensitivity analysis (30-50%)",
      TRUE           ~ "Tier D: EXCLUDE from primary model (>50%)"
    )
  )

cat("\nVariable classification:\n")
for (t in unique(overall_miss$tier)) {
  vars_in_tier <- overall_miss$variable[overall_miss$tier == t]
  pcts_in_tier <- overall_miss$pct_miss[overall_miss$tier == t]
  cat(sprintf("\n%s\n", t))
  for (i in seq_along(vars_in_tier)) {
    cat(sprintf("  %-25s (%4.1f%%)\n", vars_in_tier[i], pcts_in_tier[i]))
  }
}

# ==============================================================================
# 8. CONSTRUCT SURVEY DESIGN OBJECT
# ==============================================================================
# Pooled weight: cycle-specific weight divided by the number of cycles (9).
#   2003-2018: WTMEC2YR (MEC weight; covers MEC blood analytes in these cycles)
#   2021-2023: WTPH2YR  (phlebotomy weight; required for blood-based analytes)
# Pseudo-strata are unique across cycles; PSUs are nested within strata.

cat("\n========================================\n")
cat("SURVEY DESIGN\n")
cat("========================================\n")

cycle_years <- sort(unique(analytic$year))
n_cycles <- length(cycle_years)
cat("Cycles in analytic data:", paste(cycle_years, collapse = ", "), "\n")
cat("Number of cycles:", n_cycles, "\n")

analytic <- analytic |>
  mutate(wt_pooled = if_else(year >= 2021, wtph2yr, wtmec2yr) / n_cycles)

library(survey)
options(survey.lonely.psu = "adjust")
nhanes_design <- svydesign(ids = ~sdmvpsu, strata = ~sdmvstra, weights = ~wt_pooled,
                           nest = TRUE, data = analytic)
cat("Survey design object created successfully.\n")

# Component participation among the analytic cohort (reported in Supplement)
component_participation <- analytic |>
  group_by(year) |>
  summarise(
    n                = n(),
    blood_pressure   = round(100 * mean(!is.na(mean_sbp)), 1),
    body_measures    = round(100 * mean(!is.na(bmxbmi)), 1),
    hba1c            = round(100 * mean(!is.na(hba1c)), 1),
    lipids           = round(100 * mean(!is.na(total_chol)), 1),
    creatinine       = round(100 * mean(!is.na(creatinine)), 1),
    wbc              = round(100 * mean(!is.na(wbc)), 1),
    .groups = "drop"
  )
print(component_participation)
write.csv(component_participation, "artifacts/component_participation_by_cycle.csv", row.names = FALSE)

# Sample size and outcome prevalence by cycle (unweighted and survey-weighted)
wprev <- svyby(~stroke, ~year, nhanes_design, svymean, na.rm = TRUE)
outcome_by_cycle <- analytic |>
  group_by(year) |>
  summarise(n = n(), stroke_cases = sum(stroke == 1),
            prevalence_unweighted = round(100 * mean(stroke == 1), 2),
            sum_pooled_weight = sum(wt_pooled), .groups = "drop") |>
  left_join(tibble::tibble(year = wprev$year,
                           prevalence_weighted = round(100 * wprev$stroke, 2),
                           prevalence_weighted_se = round(100 * SE(wprev), 2)), by = "year")
print(outcome_by_cycle)
write.csv(outcome_by_cycle, "artifacts/outcome_by_cycle.csv", row.names = FALSE)

# Harmonization sanity checks by cycle (physical activity domain, age top-coding,
# alcohol skip-pattern handling) - printed so they appear in the log
harmonization_checks <- analytic |>
  group_by(year) |>
  summarise(
    n                  = n(),
    phys_active_pct    = round(100 * mean(phys_active, na.rm = TRUE), 1),
    phys_active_miss   = round(100 * mean(is.na(phys_active)), 1),
    max_age            = max(age, na.rm = TRUE),
    age_80_pct         = round(100 * mean(age == 80, na.rm = TRUE), 1),
    alcohol_zero_pct   = round(100 * mean(alcohol_drinks_per_day == 0, na.rm = TRUE), 1),
    alcohol_median     = median(alcohol_drinks_per_day, na.rm = TRUE),
    htn_med_pct        = round(100 * mean(htn_med, na.rm = TRUE), 1),
    .groups = "drop"
  )
cat("\n--- Harmonization checks by cycle ---\n")
print(as.data.frame(harmonization_checks))
write.csv(harmonization_checks, "artifacts/harmonization_checks_by_cycle.csv", row.names = FALSE)

# ==============================================================================
# 9. WEIGHTED DESCRIPTIVE STATISTICS (TABLE 1 PREVIEW)
# ==============================================================================
# Quick sanity check: weighted stroke prevalence and demographics

library(survey)

cat("\n========================================\n")
cat("WEIGHTED DESCRIPTIVE STATS (SANITY CHECK)\n")
cat("========================================\n")

# Weighted stroke prevalence
stroke_prev <- svymean(~stroke, nhanes_design, na.rm = TRUE)
cat(sprintf("\nWeighted stroke prevalence: %.2f%% (SE: %.2f%%)\n",
            coef(stroke_prev) * 100, SE(stroke_prev) * 100))

# Weighted mean age
mean_age <- svymean(~age, nhanes_design, na.rm = TRUE)
cat(sprintf("Weighted mean age: %.1f years (SE: %.1f)\n",
            coef(mean_age), SE(mean_age)))

# Weighted sex distribution
sex_dist <- svymean(~sex, nhanes_design, na.rm = TRUE)
cat("Weighted sex distribution:\n")
print(sex_dist)

# Weighted mean SBP
mean_sbp_est <- svymean(~mean_sbp, nhanes_design, na.rm = TRUE)
cat(sprintf("Weighted mean SBP: %.1f mmHg (SE: %.1f)\n",
            coef(mean_sbp_est), SE(mean_sbp_est)))

# ==============================================================================
# 10. EXPORT FOR DOWNSTREAM ANALYSIS
# ==============================================================================

# Save as parquet (efficient for Python interop with HyperImpute)
arrow::write_parquet(analytic, "data/nhanes_stroke_analytic_2003_2023.parquet")
cat("\nExported: nhanes_stroke_analytic_2003_2023.parquet\n")

# Also save as CSV for portability
write.csv(analytic, "data/nhanes_stroke_analytic_2003_2023.csv", row.names = FALSE)
cat("Exported: nhanes_stroke_analytic_2003_2023.csv\n")

# Save the survey design object for later use
saveRDS(nhanes_design, "data/nhanes_survey_design.rds")
cat("Exported: nhanes_survey_design.rds\n")

# ==============================================================================
# 11. VARIABLE CODEBOOK (for documentation / methods section)
# ==============================================================================

codebook <- tibble::tribble(
  ~variable,               ~description,                                    ~source,    ~type,         ~tier,
  "stroke",                "Self-reported stroke (MCQ160F)",                 "MCQ",      "binary",      "Outcome",
  "age",                   "Age at screening (years; top-coded at 80 in all cycles)",                "DEMO",     "continuous",  "Tier 1",
  "sex",                   "Sex (Male/Female)",                              "DEMO",     "categorical", "Tier 1",
  "race_eth",              "Race/ethnicity (5-level)",                       "DEMO",     "categorical", "Tier 1",
  "mean_sbp",              "Mean systolic BP (mmHg, avg readings 2-3)",      "BPX/BPXO", "continuous", "Tier 1",
  "mean_dbp",              "Mean diastolic BP (mmHg, avg readings 2-3)",     "BPX/BPXO", "continuous", "Tier 1",
  "htn_med",               "Currently taking antihypertensive medication",   "BPQ",      "binary",      "Tier 1",
  "hba1c",                 "Glycohemoglobin (%)",                            "GHB",      "continuous",  "Tier 1",
  "total_chol",            "Total cholesterol (mg/dL)",                      "TCHOL",    "continuous",  "Tier 1",
  "hdl_c",                 "HDL cholesterol (mg/dL)",                        "HDL",      "continuous",  "Tier 1",
  "smoking",               "Smoking status (Never/Former/Current)",          "SMQ",      "categorical", "Tier 1",
  "bmxwaist",              "Waist circumference (cm)",                       "BMX",      "continuous",  "Tier 1",
  "bmxbmi",                "Body mass index (kg/m²)",                        "BMX",      "continuous",  "Tier 1",
  "alcohol_drinks_per_day","Avg drinks per drinking day, past 12 mo (0 = non-drinker)",      "ALQ",      "continuous",  "Tier 1",
  "phys_active",           "Any moderate/vigorous physical activity",         "PAQ",      "binary",      "Tier 1",
  "triglycerides",         "Triglycerides (mg/dL)",                          "TRIGLY",   "continuous",  "Tier 2",
  "creatinine",            "Serum creatinine (mg/dL)",                       "BIOPRO",   "continuous",  "Tier 2",
  "wbc",                   "White blood cell count (1000 cells/uL)",         "CBC",      "continuous",  "Tier 2",
  "education",             "Education level (4-level)",                      "DEMO",     "categorical", "Tier 2",
  "pir",                   "Poverty-income ratio (0-5)",                     "DEMO",     "continuous",  "Tier 2",
  "sleep_hrs",             "Sleep duration (hours/night, weekdays)",          "SLQ",      "continuous",  "Tier 2",
  "diabetes_sr",           "Self-reported diabetes",                         "DIQ",      "binary",      "Tier 2",
  "chd_sr",                "Self-reported coronary heart disease",           "MCQ",      "binary",      "Tier 2"
)

# Percent missing for each candidate predictor by cycle (reported in Supplement)
missingness_by_cycle <- analytic |>
  group_by(year) |>
  summarise(across(all_of(setdiff(codebook$variable, "stroke")),
                   ~ round(100 * mean(is.na(.x)), 1)), .groups = "drop")
write.csv(missingness_by_cycle, "artifacts/missingness_by_cycle.csv", row.names = FALSE)

write.csv(codebook, "artifacts/variable_codebook.csv", row.names = FALSE)
cat("\nExported: variable_codebook.csv\n")

cat("\n============================================================\n")
cat("DATA EXTRACTION COMPLETE\n")
cat("============================================================\n")
cat("Next steps:\n")
cat("  1. Review missingness profile and tier assignments\n")
cat("  2. Export parquet to Python for HyperImpute imputation\n")
cat("  3. Run knockoff filter feature selection\n")
cat("  4. Train AutoGluon ensemble with stratified CV\n")
cat("============================================================\n")

sink()
close(.log_con)