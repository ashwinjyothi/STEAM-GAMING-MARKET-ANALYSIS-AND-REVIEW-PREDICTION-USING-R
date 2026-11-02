
# ============================================================
# PROJECT: STEAM GAMING MARKET ANALYSIS AND REVIEW PREDICTION
# Software: RStudio
# Dataset: Steam Games Dataset 2025
# ============================================================


# 1. INSTALL AND LOAD PACKAGES
# Run installation only when packages are not already installed.

packages <- c(
  "data.table",
  "dplyr",
  "ggplot2",
  "tidyr",
  "janitor",
  "caret",
  "rpart",
  "randomForest",
  "corrplot"
)

missing_packages <- packages[
  !packages %in% rownames(installed.packages())
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(data.table)
library(dplyr)
library(ggplot2)
library(tidyr)
library(janitor)
library(caret)
library(rpart)
library(randomForest)
library(corrplot)


# 2. LOAD DATASET

file_path <- "D:/LPU notes and docs/S3/Predictive analsis/Project/games_march2025_cleaned.csv"

games <- fread(file_path, na.strings = c("", "NA", "NULL"))

# Standardize column names
games <- as.data.frame(games)
games <- janitor::clean_names(games)

cat("Original dataset dimensions:\n")
print(dim(games))

cat("\nAvailable columns:\n")
print(names(games))

cat("\nFirst six rows:\n")
print(head(games))


# 3. DATA QUALITY CHECK

cat("\nMissing values in each column:\n")
print(sort(colSums(is.na(games)), decreasing = TRUE)[1:20])

cat("\nDuplicate rows:", sum(duplicated(games)), "\n")

# Remove exact duplicate rows
games <- games[!duplicated(games), ]

# Remove duplicate games by app ID, if available
if ("appid" %in% names(games)) {
  games <- games[
    !is.na(games$appid) & !duplicated(games$appid), ]
}

cat("\nDimensions after removing duplicates:\n")
print(dim(games))


# 4. FLEXIBLE COLUMN DETECTION

find_col <- function(candidates, available = names(games)) {
  found <- candidates[candidates %in% available]
  if (length(found) == 0) return(NA_character_)
  found[1]
}

price_col <- find_col(c("price", "price_usd", "final_price"))
date_col <- find_col(c("release_date", "release"))
age_col <- find_col(c("required_age", "age_requirement"))
dlc_col <- find_col(c("dlc_count", "dlc"))
genre_col <- find_col(c("genres", "genre"))
metacritic_col <- find_col(c("metacritic_score"))
positive_col <- find_col(c("positive", "positive_reviews"))
negative_col <- find_col(c("negative", "negative_reviews"))
score_col <- find_col(c("pct_pos_total", "positive_percentage"))
peak_col <- find_col(c("peak_ccu", "ccu"))
achievement_col <- find_col(c("achievements", "achievement_count"))

cat("\nDetected columns:\n")
print(list(
  price = price_col,
  release_date = date_col,
  review_percentage = score_col,
  positive_reviews = positive_col,
  negative_reviews = negative_col,
  genre = genre_col
))


# 5. DATA PREPROCESSING

# Convert numerical fields safely
to_numeric <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

numeric_cols <- c(
  price_col, age_col, dlc_col, metacritic_col,
  positive_col, negative_col, score_col,
  peak_col, achievement_col
)

numeric_cols <- unique(numeric_cols[!is.na(numeric_cols)])

for (col in numeric_cols) {
  games[[col]] <- to_numeric(games[[col]])
}

# Convert release date into a year
if (!is.na(date_col)) {
  games$release_year <- suppressWarnings(
    as.integer(substr(as.character(games[[date_col]]), 1, 4))
  )
} else {
  games$release_year <- NA_integer_
}

# Remove invalid years
games$release_year[
  games$release_year < 1970 |
    games$release_year > 2025
] <- NA

# Convert price to numeric and handle invalid prices
if (!is.na(price_col)) {
  games[[price_col]][games[[price_col]] < 0] <- NA
}

# Prepare target: positive review percentage
# Prefer the dataset's existing percentage field.
# Otherwise calculate it from positive/negative review counts.

if (!is.na(score_col)) {
  
  games$review_percentage <- games[[score_col]]
  
  # Handle percentages stored as proportions from 0 to 1
  valid_scores <- games$review_percentage[
    is.finite(games$review_percentage)
  ]
  
  if (length(valid_scores) > 0 &&
      max(valid_scores) <= 1 &&
      min(valid_scores) >= 0) {
    games$review_percentage <- games$review_percentage * 100
  }
  
} else if (!is.na(positive_col) && !is.na(negative_col)) {
  
  total_reviews <- games[[positive_col]] + games[[negative_col]]
  
  games$review_percentage <- ifelse(
    total_reviews > 0,
    100 * games[[positive_col]] / total_reviews,
    NA_real_
  )
  
} else {
  
  stop(
    "Cannot create review percentage. Check whether the dataset ",
    "contains pct_pos_total or positive and negative review counts."
  )
}

# Keep valid review percentages
games$review_percentage[
  games$review_percentage < 0 |
    games$review_percentage > 100
] <- NA

# Define classification target
games$review_class <- ifelse(
  games$review_percentage >= 70,
  "Positive",
  "Non_Positive"
)

games$review_class <- factor(
  games$review_class,
  levels = c("Positive", "Non_Positive")
)

cat("\nReview class distribution:\n")
print(table(games$review_class, useNA = "ifany"))


# 6. HANDLE MISSING VALUES

# Drop records without a valid target
games <- games[!is.na(games$review_class), ]

# Remove columns with too many missing values
# Keep columns with at least 50% non-missing data.
missing_fraction <- colMeans(is.na(games))
keep_cols <- names(missing_fraction[missing_fraction <= 0.50])
games <- games[, keep_cols, drop = FALSE]

# Check required fields again
price_col <- find_col(c("price", "price_usd", "final_price"))
genre_col <- find_col(c("genres", "genre"))

# Median imputation for selected numeric variables
impute_cols <- c(
  price_col, "release_year", age_col,
  dlc_col, metacritic_col, peak_col, achievement_col
)

impute_cols <- unique(impute_cols[!is.na(impute_cols)])
impute_cols <- intersect(impute_cols, names(games))

for (col in impute_cols) {
  x <- games[[col]]
  if (is.numeric(x) && anyNA(x)) {
    med <- median(x, na.rm = TRUE)
    if (is.finite(med)) {
      x[is.na(x)] <- med
      games[[col]] <- x
    }
  }
}

# For EDA, keep the original cleaned dataset.
# Missing values in unused text columns are not automatically
# filled because they may not be meaningful.


# 7. BASIC SUMMARY STATISTICS

cat("\nSummary statistics:\n")
print(summary(games))

cat("\nNumber of games:", nrow(games), "\n")

if (!is.na(price_col)) {
  cat("\nPrice summary:\n")
  print(summary(games[[price_col]]))
}

cat("\nReview percentage summary:\n")
print(summary(games$review_percentage))


# 8. EXPLORATORY DATA ANALYSIS (EDA)

# Plot 1: Review class distribution
p1 <- ggplot(games, aes(x = review_class, fill = review_class)) +
  geom_bar() +
  labs(
    title = "Distribution of Game Review Categories",
    x = "Review Category",
    y = "Number of Games"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p1)
ggsave("01_review_class_distribution.png", p1,
       width = 7, height = 5)

# Plot 2: Release trends
year_data <- games %>%
  filter(!is.na(release_year)) %>%
  count(release_year)

p2 <- ggplot(year_data, aes(x = release_year, y = n)) +
  geom_line() +
  geom_point(size = 1) +
  labs(
    title = "Number of Games Released by Year",
    x = "Release Year",
    y = "Number of Games"
  ) +
  theme_minimal()

print(p2)
ggsave("02_release_trends.png", p2,
       width = 8, height = 5)

# Plot 3: Price distribution
if (!is.na(price_col)) {
  p3 <- ggplot(games, aes(x = .data[[price_col]])) +
    geom_histogram(bins = 40, na.rm = TRUE) +
    coord_cartesian(
      xlim = c(0, quantile(
        games[[price_col]], 0.95, na.rm = TRUE
      ))
    ) +
    labs(
      title = "Distribution of Game Prices (Up to 95th Percentile)",
      x = "Price",
      y = "Number of Games"
    ) +
    theme_minimal()
  
  print(p3)
  ggsave("03_price_distribution.png", p3,
         width = 8, height = 5)
}

# Plot 4: Price vs review percentage
if (!is.na(price_col)) {
  p4 <- ggplot(
    games,
    aes(x = .data[[price_col]], y = review_percentage)
  ) +
    geom_point(alpha = 0.25) +
    geom_smooth(method = "lm", se = TRUE, color = "blue") +
    coord_cartesian(
      xlim = c(0, quantile(
        games[[price_col]], 0.95, na.rm = TRUE
      ))
    ) +
    labs(
      title = "Game Price vs Positive Review Percentage",
      x = "Price",
      y = "Positive Reviews (%)"
    ) +
    theme_minimal()
  
  print(p4)
  ggsave("04_price_vs_reviews.png", p4,
         width = 8, height = 5)
}

# Plot 5: Top genres
if (!is.na(genre_col)) {
  genre_data <- games %>%
    filter(!is.na(.data[[genre_col]])) %>%
    mutate(genre_text = as.character(.data[[genre_col]])) %>%
    separate_rows(genre_text, sep = "[,;]") %>%
    mutate(genre_text = trimws(genre_text)) %>%
    filter(genre_text != "") %>%
    count(genre_text, sort = TRUE) %>%
    slice_head(n = 10)
  
  p5 <- ggplot(
    genre_data,
    aes(x = reorder(genre_text, n), y = n)
  ) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Top 10 Most Common Genres",
      x = "Genre",
      y = "Number of Games"
    ) +
    theme_minimal()
  
  print(p5)
  ggsave("05_top_genres.png", p5,
         width = 8, height = 6)
}


# 9. STATISTICAL ANALYSIS

# Test 1: Is price associated with positive review percentage?
if (!is.na(price_col)) {
  stat_data <- games %>%
    filter(
      is.finite(.data[[price_col]]),
      is.finite(review_percentage)
    )
  
  if (nrow(stat_data) >= 30) {
    correlation_test <- cor.test(
      stat_data[[price_col]],
      stat_data$review_percentage,
      method = "spearman",
      exact = FALSE
    )
    
    cat("\nSpearman correlation test:\n")
    print(correlation_test)
  }
}

# Test 2: Do free and paid games have different review scores?
if (!is.na(price_col)) {
  games$price_group <- ifelse(
    games[[price_col]] == 0,
    "Free",
    ifelse(games[[price_col]] > 0, "Paid", NA)
  )
  
  test_data <- games %>%
    filter(
      !is.na(price_group),
      is.finite(review_percentage)
    )
  
  if (length(unique(test_data$price_group)) == 2 &&
      nrow(test_data) >= 30) {
    wilcox_result <- wilcox.test(
      review_percentage ~ price_group,
      data = test_data
    )
    
    cat("\nWilcoxon test: Free vs Paid games:\n")
    print(wilcox_result)
  }
}


# 10. PREPARE MACHINE LEARNING DATA

# Use predictors available in this particular CSV.
# Exclude the target and review-count fields to reduce target leakage.

predictors <- c(
  price_col, "release_year", age_col, dlc_col,
  metacritic_col, peak_col, achievement_col
)

predictors <- unique(predictors[!is.na(predictors)])
predictors <- intersect(predictors, names(games))

# Do not use the existing review percentage as a predictor
predictors <- setdiff(
  predictors,
  c(
    "review_percentage", score_col,
    positive_col, negative_col
  )
)

if (length(predictors) < 2) {
  stop(
    "Not enough usable predictors were found. ",
    "Check the detected column names and dataset schema."
  )
}

model_data <- games[, c(predictors, "review_class"), drop = FALSE]

# Keep numeric predictors only for this beginner-friendly model.
# Remove constant or entirely missing predictors.
numeric_predictors <- predictors[
  vapply(model_data[, predictors, drop = FALSE],
         is.numeric, logical(1))
]

numeric_predictors <- numeric_predictors[
  vapply(
    model_data[, numeric_predictors, drop = FALSE],
    function(x) {
      any(is.finite(x)) &&
        length(unique(x[is.finite(x)])) > 1
    },
    logical(1)
  )
]

if (length(numeric_predictors) < 2) {
  stop("Fewer than two usable numeric predictors remain.")
}

model_data <- model_data[
  complete.cases(model_data[, numeric_predictors, drop = FALSE]) &
    !is.na(model_data$review_class),
  c(numeric_predictors, "review_class"),
  drop = FALSE
]

# Cap data size to make modelling practical in RStudio.
set.seed(123)

sample_size <- min(20000, nrow(model_data))

model_data <- model_data[
  sample(seq_len(nrow(model_data)), sample_size),
  ,
  drop = FALSE
]

# Drop unused factor levels
model_data$review_class <- droplevels(model_data$review_class)

cat("\nModelling dataset dimensions:\n")
print(dim(model_data))

cat("\nModelling class distribution:\n")
print(table(model_data$review_class))

if (nlevels(model_data$review_class) != 2) {
  stop("Both review categories are required for binary classification.")
}


# 11. TRAIN-TEST SPLIT

set.seed(123)

train_index <- createDataPartition(
  model_data$review_class,
  p = 0.80,
  list = FALSE
)

train_data <- model_data[train_index, , drop = FALSE]
test_data <- model_data[-train_index, , drop = FALSE]

cat("\nTraining records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")

# Scale predictors using training data only
preprocess <- preProcess(
  train_data[, numeric_predictors, drop = FALSE],
  method = c("center", "scale")
)

train_scaled <- predict(
  preprocess,
  train_data[, numeric_predictors, drop = FALSE]
)

test_scaled <- predict(
  preprocess,
  test_data[, numeric_predictors, drop = FALSE]
)

train_scaled$review_class <- train_data$review_class
test_scaled$review_class <- test_data$review_class


# 12. MODEL 1: LOGISTIC REGRESSION

logistic_model <- glm(
  review_class ~ .,
  data = train_scaled,
  family = binomial()
)

logistic_probability <- predict(
  logistic_model,
  newdata = test_scaled,
  type = "response"
)

logistic_prediction <- factor(
  ifelse(logistic_probability >= 0.5,
         "Non_Positive", "Positive"),
  levels = levels(test_scaled$review_class)
)

logistic_cm <- confusionMatrix(
  logistic_prediction,
  test_scaled$review_class,
  positive = "Positive"
)

cat("\nLOGISTIC REGRESSION RESULTS:\n")
print(logistic_cm)


# 13. MODEL 2: DECISION TREE

tree_model <- rpart(
  review_class ~ .,
  data = train_scaled,
  method = "class",
  control = rpart.control(
    cp = 0.01,
    maxdepth = 5
  )
)

tree_prediction <- predict(
  tree_model,
  newdata = test_scaled,
  type = "class"
)

tree_prediction <- factor(
  tree_prediction,
  levels = levels(test_scaled$review_class)
)

tree_cm <- confusionMatrix(
  tree_prediction,
  test_scaled$review_class,
  positive = "Positive"
)

cat("\nDECISION TREE RESULTS:\n")
print(tree_cm)


# 14. MODEL 3: RANDOM FOREST

set.seed(123)

rf_model <- randomForest(
  review_class ~ .,
  data = train_scaled,
  ntree = 100,
  importance = TRUE
)

rf_prediction <- predict(
  rf_model,
  newdata = test_scaled,
  type = "class"
)

rf_prediction <- factor(
  rf_prediction,
  levels = levels(test_scaled$review_class)
)

rf_cm <- confusionMatrix(
  rf_prediction,
  test_scaled$review_class,
  positive = "Positive"
)

cat("\nRANDOM FOREST RESULTS:\n")
print(rf_cm)


# 15. COMPARE MODEL PERFORMANCE

get_metrics <- function(cm, model_name) {
  c(
    Model = model_name,
    Accuracy = unname(cm$overall["Accuracy"]),
    Precision = unname(cm$byClass["Pos Pred Value"]),
    Recall = unname(cm$byClass["Sensitivity"]),
    F1 = unname(cm$byClass["F1"]),
    Specificity = unname(cm$byClass["Specificity"])
  )
}

results <- as.data.frame(rbind(
  get_metrics(logistic_cm, "Logistic Regression"),
  get_metrics(tree_cm, "Decision Tree"),
  get_metrics(rf_cm, "Random Forest")
), stringsAsFactors = FALSE)

metric_cols <- c(
  "Accuracy", "Precision", "Recall", "F1", "Specificity"
)

results[metric_cols] <- lapply(
  results[metric_cols],
  as.numeric
)

cat("\nFINAL MODEL COMPARISON:\n")
print(results)

write.csv(
  results,
  "model_comparison_results.csv",
  row.names = FALSE
)

best_model <- results$Model[which.max(results$F1)]

cat("\nBest model by F1-score:", best_model, "\n")


# 16. VISUALIZE MODEL COMPARISON

results_long <- results %>%
  pivot_longer(
    cols = all_of(metric_cols),
    names_to = "Metric",
    values_to = "Score"
  )

p6 <- ggplot(
  results_long,
  aes(x = Model, y = Score, fill = Metric)
) +
  geom_col(position = "dodge") +
  labs(
    title = "Comparison of Machine Learning Models",
    x = "Model",
    y = "Performance Score"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

print(p6)

ggsave(
  "06_model_comparison.png",
  p6,
  width = 9,
  height = 5
)


# 17. RANDOM FOREST VARIABLE IMPORTANCE

varImpPlot(rf_model, main = "Random Forest Variable Importance")

png("07_variable_importance.png", width = 900, height = 650)
varImpPlot(rf_model, main = "Random Forest Variable Importance")
dev.off()


# 18. SAVE CLEANED DATA AND FINAL OUTPUTS

write.csv(
  games,
  "steam_games_cleaned.csv",
  row.names = FALSE
)

cat("\nPROJECT COMPLETED SUCCESSFULLY!\n")
cat("Files saved in the current working directory:\n")
cat("- steam_games_cleaned.csv\n")
cat("- model_comparison_results.csv\n")
cat("- 01_review_class_distribution.png\n")
cat("- 02_release_trends.png\n")
cat("- 03_price_distribution.png (if price exists)\n")
cat("- 04_price_vs_reviews.png (if price exists)\n")
cat("- 05_top_genres.png (if genres exist)\n")
cat("- 06_model_comparison.png\n")
cat("- 07_variable_importance.png\n")
