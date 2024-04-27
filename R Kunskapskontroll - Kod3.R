# Load required libraries
library(dplyr)
library(tidyr)
library(readxl)
library(ggplot2)
library(leaps)

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

# Exkludera den första kolumnen från Fordon och Bostäder
Fordon_clean <- Fordon_clean[,-1]
Bostäder_clean <- Bostäder_clean[,-1]

# Se till att kolumnnamnen är unika i Fordon_clean och Bostäder_clean
# Du kan till exempel lägga till ett prefix till varje kolumnnamn i Bostäder_clean
names(Bostäder_clean) <- paste("Bostäder", names(Bostäder_clean), sep="_")

# Kombinera 'Fordon_clean' och 'Bostäder' till en enda dataframe
# Antag att 'År_2013', 'År_2014', etc., är kolumnerna i 'Bostäder'
combined_data <- cbind(Fordon_clean, Bostäder_clean)

# Steg 2 och 3: Modellbyggnad och val av den bästa modellen för varje delmängdsstorlek
# Ersätt 'Response' med den kolumn du vill förutsäga, t.ex. 'År_2013'
best_subsets <- regsubsets(År.2013 ~ ., data = combined_data, nvmax = ncol(combined_data) - 1, method = "exhaustive")

# Steg 4: Val av den bästa modellen bland alla delmängder
subset_results <- summary(best_subsets)
best_model <- which.min(subset_results$bic)

# Visa den bästa modellen
print(subset_results$which[best_model, ])