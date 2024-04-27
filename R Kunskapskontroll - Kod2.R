# Load required libraries
library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)

# Läs in data
Fordon <- read_excel("C:/Users/User/Downloads/Fordon.xlsx")
Bostäder <- read_excel("C:/Users/User/Downloads/Bostäder.xlsx")

# Ersätt NA-värden med medianen för varje kolumn i Fordon och Bostäder datasets
Fordon_cleaning <- Fordon
Fordon_cleaned <- lapply(Fordon_cleaning, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
Bostäder_cleaning <- Bostäder
Bostäder_cleaned <- lapply(Bostäder_cleaning, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Konvertera listan till en dataframe
Fordon_clean <- data.frame(Fordon_cleaned)
Bostäder_clean <- data.frame(Bostäder_cleaned)

# Skapa matriser utan 'Region'-kolumnen
Fordon_matrix <- as.matrix(Fordon_clean[,-1])
Bostäder_matrix <- as.matrix(Bostäder_clean[,-1])

# Funktion för att utföra linjär regression för en rad (region)
run_regression_per_region <- function(fordon_row, bostader_row) {
  lm_result <- lm(bostader_row ~ fordon_row)
  return(lm_result)
}

# Använd 'mapply()' för att köra 'run_regression_per_region' för varje rad
regression_results <- mapply(run_regression_per_region, split(Fordon_matrix, row(Fordon_matrix)), split(Bostäder_matrix, row(Bostäder_matrix)), SIMPLIFY = FALSE)

# Skapa en data.frame för att visa resultat
results_df <- data.frame(Region = Fordon_clean$Region, Intercept = rep(NA, length(Fordon_clean$Region)), Slope = rep(NA, length(Fordon_clean$Region)), t_value = rep(NA, length(Fordon_clean$Region)), p_value = rep(NA, length(Fordon_clean$Region)))

# Loopa igenom varje regression_result och extrahera koefficienter, t- och p-värden
for (i in seq_along(regression_results)) {
  results_df$Intercept[i] <- regression_results[[i]]$coefficients[1]
  results_df$Slope[i] <- regression_results[[i]]$coefficients[2]
  results_df$t_value[i] <- summary(regression_results[[i]])$coefficients[2, "t value"]
  results_df$p_value[i] <- summary(regression_results[[i]])$coefficients[2, "Pr(>|t|)"]
}

# Skapa en tom lista för att lagra plottarna
plots_list <- list()

# Uppdatera plottarna för att inkludera en streckad regressionslinje
for (i in seq_along(regression_results)) {
  region <- results_df$Region[i]
  model <- regression_results[[i]]
  
  # Skapa en data.frame med de faktiska värdena och de förutsagda värdena
  actual_vs_predicted <- data.frame(
    Actual = Bostäder_matrix[i, ],
    Predicted = predict(model, newdata = data.frame(fordon_row = Fordon_matrix[i, ]))
  )
  
  # Sortera data.frame baserat på de faktiska värdena för att linjen ska bli korrekt
  actual_vs_predicted <- actual_vs_predicted[order(actual_vs_predicted$Actual), ]
  
  # Skapa plotten med en linje som förbinder punkterna och en streckad regressionslinje
  plot <- ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
    geom_point() +
    geom_line() +
    geom_smooth(method = "lm", se = FALSE, col = "blue", linetype = "dashed") + # Lägg till denna rad
    labs(title = paste("Linjärt Samband för Region", region),
         x = "Faktiska Värden",
         y = "Förutsagda Värden") +
    theme_minimal()
  
  # Lägg till plotten i listan
  plots_list[[region]] <- plot
}

# Visa plotten för första regionen som ett exempel
print(plots_list[[results_df$Region[1]]])

# Loopa igenom varje regression_result och extrahera koefficienter, t- och p-värden
for (i in seq_along(regression_results)) {
  results_df$Intercept[i] <- regression_results[[i]]$coefficients[1]
  results_df$Slope[i] <- regression_results[[i]]$coefficients[2]
  results_df$t_value[i] <- summary(regression_results[[i]])$coefficients[2, "t value"]
  results_df$p_value[i] <- summary(regression_results[[i]])$coefficients[2, "Pr(>|t|)"]
  
  # Skriv ut sammanfattningen för varje modell
  cat("Sammanfattning för Region", results_df$Region[i], ":\n")
  print(summary(regression_results[[i]]))
  cat("\n")
}