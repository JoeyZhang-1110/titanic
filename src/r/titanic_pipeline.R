library(readr)
library(dplyr)
library(caret)
library(stringr)
library(tidyr)

log_msg <- function(msg) cat(paste0("[INFO] ", msg, "\n"))

DATA_DIR <- "src/data"

log_msg(paste("Expecting Titanic CSVs in:", normalizePath(DATA_DIR)))
train <- read_csv(file.path(DATA_DIR, "train.csv"))
test  <- read_csv(file.path(DATA_DIR, "test.csv"))

log_msg(paste("Train shape:", paste(dim(train), collapse = " x ")))
log_msg(paste("Test  shape:", paste(dim(test), collapse = " x ")))

engineer_features <- function(df) {
  log_msg("Engineering features...")
  
  df <- df %>%
    mutate(
      Title = str_extract(Name, ",\\s*([^\\.]+)\\.") %>%
        str_replace_all(",\\s*|\\.", "") %>%
        str_trim(),
      CabinFirstLetter = ifelse(is.na(Cabin), NA, substr(Cabin, 1, 1)),
      FamilySize = coalesce(SibSp, 0) + coalesce(Parch, 0) + 1
    )
  
  keep <- c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
            "Fare", "Embarked", "Title", "CabinFirstLetter", "FamilySize")
  keep <- intersect(keep, names(df))
  df <- df[, keep]
  log_msg(paste("Feature columns now:", paste(keep, collapse = ", ")))
  return(df)
}

train_fe <- engineer_features(train)
test_fe  <- engineer_features(test)

if (!"Survived" %in% names(train_fe)) {
  stop("[ERROR] 'Survived' column not found in train.csv.")
}

train_fe$Sex <- as.factor(train_fe$Sex)
train_fe$Embarked <- as.factor(train_fe$Embarked)
train_fe$Title <- as.factor(train_fe$Title)
train_fe$CabinFirstLetter <- as.factor(train_fe$CabinFirstLetter)

log_msg("Fitting logistic regression on full training set...")
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare +
               Embarked + Title + CabinFirstLetter + FamilySize,
             data = train_fe, family = binomial())

pred_train <- ifelse(predict(model, type = "response") > 0.5, 1, 0)
train_acc <- mean(pred_train == train_fe$Survived)
log_msg(paste("TRAIN accuracy (on full train.csv):", round(train_acc, 4)))

log_msg("Generating predictions on test.csv...")
pred_test <- ifelse(predict(model, newdata = test_fe, type = "response") > 0.5, 1, 0)

out <- data.frame(
  PassengerId = test$PassengerId,
  Survived = pred_test
)
write_csv(out, "r_submission.csv")

log_msg("Saved predictions to: r_submission.csv")
log_msg("Note: test.csv has no 'Survived' column; skipping accuracy check as instructed.")
