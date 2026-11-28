#!/usr/bin/env Rscript

required_packages <- c(
  "brms", "cmdstanr", "dplyr", "lubridate", "mgcv",
  "posterior", "readr", "stringr", "tidyr"
)
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
  library(dplyr)
  library(lubridate)
  library(mgcv)
  library(readr)
  library(stringr)
  library(tidyr)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) {
  stop("Run this script with Rscript hierarchical_model/validate_time_holdout.R")
}
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
root <- normalizePath(file.path(dirname(script_path), ".."))
output_dir <- file.path(root, "outputs", "hourly-time-holdout")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cmdstan_path <- tryCatch(
  cmdstanr::cmdstan_path(),
  error = function(error) {
    stop("CmdStan is required; install it with cmdstanr::install_cmdstan().", call. = FALSE)
  }
)
if (!dir.exists(cmdstan_path)) {
  stop("CmdStan path does not exist: ", cmdstan_path, call. = FALSE)
}

key <- c("year", "month", "day", "time")
raw_files <- list.files(
  file.path(root, "data", "raw"),
  pattern = "\\.csv$",
  full.names = TRUE
)
if (!length(raw_files)) {
  stop("No raw CSV files found under data/raw.", call. = FALSE)
}
raw_data <- lapply(
  raw_files,
  readr::read_csv,
  na = c("", "NA", "-"),
  show_col_types = FALSE
)

clean_names <- function(data) {
  dplyr::rename(
    data,
    station = `Observation station`,
    year = Year,
    month = Month,
    day = Day,
    time = `Time [Local time]`
  )
}

one_column <- function(pattern, columns, file) {
  matches <- grep(pattern, columns, value = TRUE)
  if (length(matches) != 1L) {
    stop(
      "Expected exactly one column matching ", pattern, " in ", file,
      "; found ", length(matches), ".",
      call. = FALSE
    )
  }
  matches[[1]]
}

is_no2_file <- function(data) {
  any(grepl("^Nitrogen dioxide", names(data)))
}
is_weather_file <- function(data) {
  any(grepl("^Maximum wind", names(data)))
}

no2_files <- which(vapply(raw_data, is_no2_file, logical(1)))
weather_files <- which(vapply(raw_data, is_weather_file, logical(1)))
if (!length(no2_files) || !length(weather_files)) {
  stop("Raw files must include NO2 and weather observations.", call. = FALSE)
}

nitrogen <- dplyr::bind_rows(lapply(no2_files, function(index) {
  data <- clean_names(raw_data[[index]])
  column <- one_column("^Nitrogen dioxide", names(data), raw_files[[index]])
  dplyr::transmute(
    data,
    station = as.character(station),
    year,
    month,
    day,
    time,
    no2 = as.numeric(.data[[column]])
  )
}))

weather <- dplyr::bind_rows(lapply(weather_files, function(index) {
  data <- clean_names(raw_data[[index]])
  columns <- c(
    wind = one_column("^Maximum wind", names(data), raw_files[[index]]),
    precip = one_column("^Precipitation", names(data), raw_files[[index]]),
    temp = one_column("^Average temp", names(data), raw_files[[index]])
  )
  dplyr::transmute(
    data,
    year,
    month,
    day,
    time,
    wind = as.numeric(.data[[columns[["wind"]]]]),
    precip = as.numeric(.data[[columns[["precip"]]]]),
    temp = as.numeric(.data[[columns[["temp"]]]])
  )
}))

weather_hourly <- weather |>
  dplyr::summarise(
    dplyr::across(c(wind, precip, temp), \(x) mean(x, na.rm = TRUE)),
    .by = dplyr::all_of(key)
  )
if (anyDuplicated(weather_hourly[key])) {
  stop("Regional weather keys are not unique; refusing a many-to-many join.", call. = FALSE)
}

joined <- dplyr::left_join(nitrogen, weather_hourly, by = key)
hour_text <- stringr::str_sub(as.character(joined$time), 1, 2)
if (anyNA(hour_text) || any(!grepl("^[0-9]{2}$", hour_text))) {
  stop("A source time label does not begin with a two-digit hour.", call. = FALSE)
}
joined$hour_label <- as.integer(hour_text)
if (any(joined$hour_label < 0 | joined$hour_label > 23)) {
  stop("A source time label is outside the 00:00–23:00 range.", call. = FALSE)
}
if (any(joined$year != 2024)) {
  stop("This validation script is configured for 2024 source data only.", call. = FALSE)
}
joined$datetime <- lubridate::make_datetime(
  as.integer(joined$year),
  as.integer(joined$month),
  as.integer(joined$day),
  joined$hour_label,
  tz = "UTC"
)
if (anyNA(joined$datetime)) {
  stop("Invalid calendar date in the source files.", call. = FALSE)
}

data <- joined |>
  tidyr::drop_na(no2, wind, precip, temp) |>
  dplyr::filter(no2 > 0) |>
  dplyr::mutate(
    date = as.Date(datetime),
    hour = lubridate::hour(datetime),
    yday = lubridate::yday(datetime),
    weekend = as.numeric(lubridate::wday(datetime, week_start = 1) >= 6),
    log_no2 = log(no2)
  ) |>
  dplyr::arrange(station, datetime)

if (any(!is.finite(data$log_no2)) || any(!is.finite(data$hour))) {
  stop("Prepared model data contain nonfinite values.", call. = FALSE)
}
station_levels <- sort(unique(data$station))
if (length(station_levels) != 4L) {
  stop("Expected the four NO2 stations used in the report.", call. = FALSE)
}
data$station <- factor(data$station, levels = station_levels)

split_date <- as.Date("2024-10-01")
train_raw <- dplyr::filter(data, date < split_date)
test_raw <- dplyr::filter(data, date >= split_date)
if (!nrow(train_raw) || !nrow(test_raw)) {
  stop("The train or Q4 test partition is empty.", call. = FALSE)
}
if (max(train_raw$date) >= min(test_raw$date)) {
  stop("Training dates must precede all test dates.", call. = FALSE)
}
if (!setequal(unique(test_raw$station), station_levels)) {
  stop("Every test station must also occur in training.", call. = FALSE)
}

centers <- c(
  temp = mean(train_raw$temp),
  log_wind = mean(log1p(train_raw$wind))
)
scales <- c(
  temp = stats::sd(train_raw$temp),
  log_wind = stats::sd(log1p(train_raw$wind))
)
if (!all(is.finite(centers)) || !all(is.finite(scales)) || any(scales <= 0)) {
  stop("Training-only weather transforms are invalid.", call. = FALSE)
}

apply_training_transforms <- function(frame) {
  frame |>
    dplyr::mutate(
      temp_z = (temp - centers[["temp"]]) / scales[["temp"]],
      lwind_z = (log1p(wind) - centers[["log_wind"]]) / scales[["log_wind"]]
    )
}
train <- apply_training_transforms(train_raw)
test <- apply_training_transforms(test_raw)

pooled_formula <- brms::bf(
  log_no2 ~ temp_z * lwind_z + weekend +
    s(hour, bs = "cc", k = 10) + s(yday, bs = "cc", k = 10),
  family = gaussian(),
  center = FALSE
)
hierarchical_formula <- brms::bf(
  log_no2 ~ temp_z * lwind_z + weekend + (1 + weekend | station) +
    s(hour, bs = "cc", k = 10) + s(yday, bs = "cc", k = 10),
  family = gaussian(),
  center = FALSE
)

common_priors <- c(
  brms::prior(normal(log(10), 1), class = "b", coef = "Intercept"),
  brms::prior(normal(0, 1), class = "b"),
  brms::prior(normal(0, 1), class = "sds"),
  brms::prior(normal(0, 2), class = "sigma", lb = 0)
)
hierarchical_priors <- c(
  common_priors,
  brms::prior(normal(0, 0.5), class = "sd", coef = "Intercept", group = "station"),
  brms::prior(normal(0, 0.5), class = "sd", coef = "weekend", group = "station"),
  brms::prior(lkj(2), class = "cor")
)

fit_model <- function(formula, prior, seed, model_name) {
  message("Fitting ", model_name, " on ", nrow(train), " January–September rows.")
  fit <- brms::brm(
    formula = formula,
    prior = prior,
    data = train,
    backend = "cmdstanr",
    chains = 4,
    cores = 4,
    iter = 2000,
    warmup = 1000,
    seed = seed,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    knots = list(hour = c(0, 23), yday = c(1, 366)),
    refresh = 0
  )
  saveRDS(fit, file.path(output_dir, paste0(model_name, "_fit.rds")))
  capture.output(
    print(summary(fit)),
    file = file.path(output_dir, paste0(model_name, "_summary.txt"))
  )
  fit
}

pooled_fit <- fit_model(
  pooled_formula, common_priors, 20261128, "pooled"
)
hierarchical_fit <- fit_model(
  hierarchical_formula, hierarchical_priors, 20261129, "hierarchical"
)

summarize_ratio <- function(ratios, model, level) {
  if (!length(ratios) || !all(is.finite(ratios)) || any(ratios <= 0)) {
    stop("Weekend-effect ratios must be finite and positive.", call. = FALSE)
  }
  reductions <- 100 * (1 - ratios)
  ratio_interval <- stats::quantile(ratios, c(0.025, 0.975), names = FALSE)
  reduction_interval <- stats::quantile(reductions, c(0.025, 0.975), names = FALSE)
  data.frame(
    model = model,
    level = level,
    ratio_median = stats::median(ratios),
    ratio_lower_95 = ratio_interval[[1]],
    ratio_upper_95 = ratio_interval[[2]],
    reduction_percent_median = stats::median(reductions),
    reduction_percent_lower_95 = reduction_interval[[1]],
    reduction_percent_upper_95 = reduction_interval[[2]]
  )
}

weekend_contrast <- function(fit, profile, include_station_effects) {
  weekday <- profile
  weekend <- profile
  weekday$weekend <- 0
  weekend$weekend <- 1
  re_formula <- if (include_station_effects) NULL else NA
  eta_weekday <- brms::posterior_linpred(
    fit, newdata = weekday, re_formula = re_formula
  )
  eta_weekend <- brms::posterior_linpred(
    fit, newdata = weekend, re_formula = re_formula
  )
  exp(eta_weekend - eta_weekday)
}

reference_rows <- train |>
  dplyr::arrange(station, datetime) |>
  dplyr::distinct(station, .keep_all = TRUE) |>
  dplyr::select(station, temp_z, lwind_z, hour, yday)
population_profile <- reference_rows[1, , drop = FALSE]

effect_rows <- list()
for (entry in list(
  list(name = "pooled", fit = pooled_fit),
  list(name = "hierarchical", fit = hierarchical_fit)
)) {
  name <- entry$name
  fit <- entry$fit
  population_ratio <- weekend_contrast(
    fit, population_profile, include_station_effects = FALSE
  )
  effect_rows[[length(effect_rows) + 1L]] <- summarize_ratio(
    as.numeric(population_ratio),
    name,
    "population"
  )
  if (name == "hierarchical") {
    station_ratios <- weekend_contrast(
      fit, reference_rows, include_station_effects = TRUE
    )
    for (column in seq_len(ncol(station_ratios))) {
      effect_rows[[length(effect_rows) + 1L]] <- summarize_ratio(
        station_ratios[, column],
        name,
        as.character(reference_rows$station[[column]])
      )
    }
  }
}
weekend_effects <- dplyr::bind_rows(effect_rows)
readr::write_csv(weekend_effects, file.path(output_dir, "weekend_effects.csv"))

predict_model <- function(fit, model_name, seed) {
  set.seed(seed)
  log_draws <- brms::posterior_predict(
    fit,
    newdata = test,
    re_formula = NULL,
    ndraws = 2000
  )
  if (!is.matrix(log_draws) || ncol(log_draws) != nrow(test)) {
    stop("Unexpected posterior_predict output shape for ", model_name, ".", call. = FALSE)
  }
  concentration_draws <- exp(log_draws)
  if (!all(is.finite(concentration_draws)) || any(concentration_draws <= 0)) {
    stop("Nonfinite predictive concentrations for ", model_name, ".", call. = FALSE)
  }
  intervals <- apply(
    concentration_draws,
    2,
    stats::quantile,
    probs = c(0.025, 0.975),
    names = FALSE
  )
  data.frame(
    model = model_name,
    date = test$date,
    month = format(test$date, "%Y-%m"),
    station = as.character(test$station),
    observed = test$no2,
    predicted_mean = colMeans(concentration_draws),
    lower_95 = intervals[1, ],
    upper_95 = intervals[2, ]
  )
}

predictions <- dplyr::bind_rows(
  predict_model(pooled_fit, "pooled", 20261130),
  predict_model(hierarchical_fit, "hierarchical", 20261131)
)

summarize_predictions <- function(frame, groups = character()) {
  group_columns <- c("model", groups)
  frame |>
    dplyr::group_by(dplyr::across(dplyr::all_of(group_columns))) |>
    dplyr::summarise(
      n = dplyr::n(),
      mae = mean(abs(predicted_mean - observed)),
      rmse = sqrt(mean((predicted_mean - observed)^2)),
      predictive_coverage_95 = mean(observed >= lower_95 & observed <= upper_95),
      mean_interval_width_95 = mean(upper_95 - lower_95),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      scope = if (length(groups)) paste(groups, collapse = "+") else "overall"
    )
}

metrics <- dplyr::bind_rows(
  summarize_predictions(predictions),
  summarize_predictions(predictions, "month"),
  summarize_predictions(predictions, "station"),
  summarize_predictions(predictions, c("station", "month"))
) |>
  dplyr::relocate(scope, .before = model)

readr::write_csv(predictions, file.path(output_dir, "predictions.csv"))
readr::write_csv(metrics, file.path(output_dir, "metrics.csv"))
capture.output(sessionInfo(), file = file.path(output_dir, "session-info.txt"))
writeLines(
  c(
    "Exploratory forward-time validation; not an untouched final test.",
    "Train: dates before 2024-10-01. Test: 2024-10-01 through 2024-12-31.",
    "All stations on each date are kept in the same partition.",
    "Weather transforms are estimated on training rows and reused unchanged on test rows.",
    "Both models use the same rows, fixed effects, spline basis, and half-normal sigma prior.",
    "Weekend labels and spline boundaries match the hourly report; timestamp semantics remain unresolved."
  ),
  file.path(output_dir, "README.txt")
)

message("Holdout results written to ", output_dir)
print(metrics)
print(weekend_effects)
