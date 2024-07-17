setwd("C:/Users/zl23n/OneDrive - Florida State University/Graduate-FSU/Research/2024 Airport TR-B/airport_code/Airport Code 0715/summary_stat/PHX")
data <- read.csv("PHX_Annual.csv")
colnames(data) <- c("year", "passenger")
head(data)


data$passenger = data$passenger / 1e7


# Create lagged column
data$passenger_next <- c(data$passenger[-1], NA)
head(data)

data <- data[-1, ]
head(data)


data <- subset(data, year < 2019)
tail(data)

data$y <- data$passenger_next / data$passenger
data$x <- data$passenger


model <- lm(y ~ x, data = data)
summary(model)

plot(model)




# Set up the layout for a 2x2 grid of plots
pdf("PHX diagnostic_plots.pdf")  # Open a PDF file to save the plots

par(mfrow = c(2, 2))  # Arrange plots in a 2x2 grid

# Plot 1: Residuals vs Fitted values
plot(model, which = 1)

# Plot 2: Normal Q-Q plot of residuals
plot(model, which = 2)

# Plot 3: Scale-location plot (sqrt(| residuals |) vs Fitted)
plot(model, which = 3)

# Plot 4: Residuals vs Leverage plot
plot(model, which = 5)

dev.off()  # Close the PDF file



