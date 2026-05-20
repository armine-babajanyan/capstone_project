# List of required R packages
required_packages <- c(
  "tidyverse", "lubridate", "skimr", "janitor", "haven", 
  "readxl", "data.table", "rmarkdown", "knitr", "effsize", 
  "car", "gridExtra", "RColorBrewer", "scales"
)

# Identify missing packages
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

# Install missing ones
if (length(missing_packages)) {
  cat("Installing missing R packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
} else {
  cat("All R packages are already installed.\n")
}
