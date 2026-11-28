#!/usr/bin/env Rscript

required_packages <- c("bayesplot", "brms", "cmdstanr", "ggplot2", "loo")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Install the required R packages before running this script: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(brms)
  library(ggplot2)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) {
  stop("Run this script with Rscript hierarchical_model/plot_posterior_checks.R")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
root <- normalizePath(file.path(dirname(script_path), ".."))

fit_paths <- c(
  pooled = file.path(root, "hierarchical_model", "saved_models", "pooled_model.rds"),
  hierarchical = file.path(
    root, "hierarchical_model", "saved_models", "hierarchical_model.rds"
  )
)
selected <- commandArgs(trailingOnly = TRUE)
if (!length(selected)) {
  selected <- c("pooled", "hierarchical")
}
unknown <- setdiff(selected, names(fit_paths))
if (length(unknown)) {
  stop(
    "Unknown fit name(s): ", paste(unknown, collapse = ", "),
    ". Choose pooled or hierarchical.",
    call. = FALSE
  )
}

output_dir <- file.path(root, "reports", "figures", "ppc")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

save_ppc <- function(name, fit, type, suffix, ndraws = 100, ...) {
  plot <- brms::pp_check(fit, type = type, ndraws = ndraws, ...)
  path <- file.path(output_dir, paste0(name, "_", suffix, ".png"))
  ggplot2::ggsave(path, plot = plot, width = 8, height = 5, dpi = 300)
  message("Saved ", path)
}

for (name in selected) {
  path <- fit_paths[[name]]
  if (!file.exists(path)) {
    stop("Saved fit not found: ", path, call. = FALSE)
  }
  fit <- readRDS(path)
  if (!inherits(fit, "brmsfit")) {
    stop("Saved object is not a brmsfit: ", path, call. = FALSE)
  }

  save_ppc(name, fit, "dens_overlay", "density")
  save_ppc(name, fit, "ecdf_overlay", "ecdf")
  save_ppc(name, fit, "loo_pit_overlay", "loo_pit", ndraws = 300)
  for (stat in c("mean", "sd", "max")) {
    save_ppc(
      name, fit, "stat_grouped", paste0("station_", stat),
      ndraws = 200, group = "station", stat = stat
    )
  }
  save_ppc(
    name, fit, "stat_grouped", "station_q95",
    ndraws = 200,
    group = "station",
    stat = function(y) unname(stats::quantile(y, 0.95))
  )
}
