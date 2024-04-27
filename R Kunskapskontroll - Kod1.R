# Load required libraries
library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)

# Read data
Fordon <- read_excel("C:/Users/User/Downloads/Fordon.xlsx")
Bostäder <- read_excel("C:/Users/User/Downloads/Bostäder.xlsx")

# Clean data: att ta bort den första raden och den första kolumnen från 'Fordon' och 
# 'Bostäder' gör att merge_data inte känner igen kolumnen. Error in `left_join()`:
# ! Join columns in `x` must be present in the data.

# Att ta bort rader med NA-värden kan leda till att fler rader blir borttagna ur ena 
# tabellen än den andra vilket gör att de inte kan sammanfogas.
# t.ex. Fordon_clean <- na.omit(Fordon), Bostäder_clean <- na.omit(Bostäder)

# Replace missing values with median for each column in Fordon and Bostäder datasets
Fordon_cleaning <- Fordon
Fordon_cleaned <- lapply(Fordon_cleaning, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

Bostäder_cleaning <- Bostäder
Bostäder_cleaned <- lapply(Bostäder_cleaning, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Omvandla listan till en dataram
Fordon_clean <- data.frame(Fordon_cleaned)
Bostäder_clean <- data.frame(Bostäder_cleaned)

# Kontrollera strukturen på den nya dataramen
str(Fordon_clean)

# Initialize merged_data with the Region column
merged_data <- data.frame(Region = intersect(Fordon_clean[[1]], Bostäder_clean[[1]]))

for (year_col in 2:ncol(Fordon_clean)) {
  year <- colnames(Fordon_clean)[year_col]
  
  # Extract columns for the current year
  subset_Fordon <- Fordon_clean[, c(1, year_col)]
  subset_Bostäder <- Bostäder_clean[, c(1, year_col)]
  
  # Merge the subsets for the current year
  merged_year <- left_join(subset_Fordon, subset_Bostäder, by = "Region")
  
  # Rename the value columns to indicate their source
  colnames(merged_year) <- c("Region", paste0("Value_Fordon_", year), paste0("Value_Bostäder_", year))
  
  # Merge the current year's data with the existing merged_data
  merged_data <- merged_data %>% left_join(merged_year, by = "Region")
}

# Linear Regression for each year
regression_models <- list()
plots <- list()

# Extract years from column names
years <- unique(gsub("Value_Fordon_|Value_Bostäder_", "", colnames(merged_data)[-1]))

for (year in years) {
  ford_col <- paste0("Value_Fordon_", year)
  bost_col <- paste0("Value_Bostäder_", year)
  
  # Fit linear regression model
  model <- lm(merged_data[[bost_col]] ~ merged_data[[ford_col]], data = merged_data)
  regression_models[[year]] <- model
  
  # Plot data and regression line
  plot <- ggplot(merged_data, aes_string(x = ford_col, y = bost_col)) +
    geom_point() +
    geom_smooth(method = "lm", col = "blue") +
    theme_minimal() +
    labs(title = paste("Linear Regression for Year", year), 
         x = "Fordon",
         y = "Bostäder")
  
  plots[[year]] <- plot
  
  # Extract coefficients
  coef_data <- coef(summary(model))
  ford_coef_row <- grep(ford_col, rownames(coef_data))
  
  # Extract estimated coefficient and standard error
  slope_estimate <- coef_data[ford_coef_row, "Estimate"]
  slope_se <- coef_data[ford_coef_row, "Std. Error"]
  
  # Calculate t-value
  t_value <- slope_estimate / slope_se
  
  # Calculate degrees of freedom
  df <- df.residual(model)
  
  # Compute p-value
  p_value <- 2 * pt(abs(t_value), df = df, lower.tail = FALSE)  # Two-tailed test
  
  # Print t-value and p-value
  cat("Summary for Year", year, ":\n")
  print(summary(model))
  print(plots[[year]])
  print(paste("t-value:", t_value))
  print(paste("p-value:", p_value))
}