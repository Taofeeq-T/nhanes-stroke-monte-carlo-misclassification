################################################################################
# NHANES Stroke Prediction — Missingness Heatmap (Figure for Publication)
# Produces a variable × cycle heatmap of missingness rates
################################################################################

library(systemfonts)  
library(textshaping)
library(ragg)
library(svglite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)
library(scales)

# ==============================================================================
# 1. Load Data and Codebook
# ==============================================================================

analytic <- read.csv("data/nhanes_stroke_analytic_2003_2023.csv", stringsAsFactors = FALSE)
codebook <- read.csv("artifacts/variable_codebook.csv", stringsAsFactors = FALSE)

cat("Loaded analytic dataset:", nrow(analytic), "rows x", ncol(analytic), "columns\n")

# Prepare predictor info from codebook (Single Source of Truth)
predictor_info <- codebook |>
  filter(tier != "Outcome") |>
  rename(display_name = description, domain = source)

# Clean labels: strip units in parentheses for cleaner plot layout
predictor_info$display_name <- gsub(" \\(.*\\)", "", predictor_info$display_name)

# ==============================================================================
# 2. Compute Missingness
# ==============================================================================

miss_by_cycle <- analytic |>
  group_by(year) |>
  summarise(
    across(
      all_of(predictor_info$variable),
      ~round(mean(is.na(.)) * 100, 1),
      .names = "{.col}"
    ),
    n_cycle = n(),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = -c(year, n_cycle),
    names_to = "variable",
    values_to = "pct_missing"
  ) |>
  left_join(predictor_info, by = "variable")

# Order variables by domain (top to bottom)
miss_by_cycle$display_name <- factor(
  miss_by_cycle$display_name,
  levels = rev(unique(predictor_info$display_name))
)

# ==============================================================================
# 3. Figure: Missingness Heatmap
# ==============================================================================

# Calculate domain boundary lines dynamically
domain_counts <- predictor_info |>
  count(domain) |>
  mutate(pos = cumsum(n) + 0.5) |>
  pull(pos)

p <- ggplot(miss_by_cycle, aes(x = factor(year), y = display_name, fill = pct_missing)) +
  geom_tile(color = "white", linewidth = 0.4) +
  
  # Labels: "0" for truly complete, "<1" for near-zero, integer for rest
  geom_text(
    aes(
      label = case_when(
        pct_missing == 0 ~ "0",
        pct_missing < 1  ~ "<1",
        TRUE             ~ sprintf("%.0f", pct_missing)
      ),
      color = pct_missing > 40
    ),
    size = 2.8
  ) +
  scale_color_manual(
    values = c("FALSE" = "grey20", "TRUE" = "white"),
    guide = "none"
  ) +
  
  scale_fill_viridis_c(
    name = "Missing (%)",
    option = "inferno",
    direction = -1,
    limits = c(0, 100),
    breaks = c(0, 5, 30, 50, 75, 100),
    labels = c("0%", "5%", "30%", "50%", "75%", "100%"),
    na.value = "grey90",
    guide = guide_colorbar(
      barheight = unit(6, "cm"),
      ticks.linewidth = 1.5,
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  
  # Dynamic domain separators
  geom_hline(
    yintercept = domain_counts[-length(domain_counts)],
    color = "grey40", linewidth = 0.6, linetype = "solid"
  ) +
  
  labs(
    title = "Predictor Missingness Across NHANES Cycles (2003-2023)",
    subtitle = "Decision Tiers: A (<5%) impute freely | B (5-30%) impute + verify diagnostics | C (30-50%) sensitivity analysis | D (>50%) exclude",
    x = "NHANES Cycle (Start Year)",
    y = NULL,
    caption = "Source: NHANES 2003-2023 via nhanesdata R package. N reflects adults >=20 with valid stroke response."
  ) +
  
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, hjust = 0),
    plot.subtitle = element_text(size = 9, color = "grey40", hjust = 0),
    plot.caption  = element_text(size = 7, color = "grey50", hjust = 0),
    axis.text.x   = element_text(size = 9),
    axis.text.y   = element_text(size = 9),
    legend.position  = "right",
    legend.text      = element_text(size = 8, margin = margin(l = 2)),
    legend.title     = element_text(size = 10, face = "bold"),
    panel.grid    = element_blank(),
    plot.margin   = margin(10, 15, 10, 10)
  )

# ==============================================================================
# 4. Save Outputs
# ==============================================================================

ggsave("artifacts/fig_missingness_heatmap.png", p, width = 12, height = 8, dpi = 300)
ggsave("artifacts/fig_missingness_heatmap.pdf", p, width = 12, height = 8)
ggsave("artifacts/fig_missingness_heatmap.svg", p, width = 12, height = 8)

# Export tier classification for HyperImpute script
overall_miss <- analytic |>
  summarise(across(all_of(predictor_info$variable), ~round(mean(is.na(.)) * 100, 1))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "overall_pct") |>
  left_join(predictor_info, by = "variable") |>
  mutate(
    final_tier = case_when(
      overall_pct < 5  ~ "A",
      overall_pct < 30 ~ "B",
      overall_pct < 50 ~ "C",
      TRUE             ~ "D"
    )
  )

write.csv(overall_miss, "artifacts/missingness_tier_classification.csv", row.names = FALSE)

cat("Success! Heatmap and Tier CSV are saved.\n")