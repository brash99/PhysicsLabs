# data on student-faculty ratios at CNU
# read data from a TSV file - year, ratio, uncertainty in ratio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Read the data from TSV file
data = pd.read_csv('student_faculty_ratio.tsv', sep='\t', header=None, names=['Year', 'Ratio', 'Uncertainty'])
# Display the first few rows of the dataframe
print(data.head())

# Extract year, ratio, and uncertainty
years = data['Year']
ratios = data['Ratio']
uncertainties = data['Uncertainty']

# Plotting the data with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(years, ratios, yerr=uncertainties, fmt='o', ecolor='r', capsize=5, label='Student-Faculty Ratio')
plt.title('Student-Faculty Ratio at CNU Over Years')
plt.xlabel('Year')
plt.ylabel('Student-Faculty Ratio')
plt.grid(True)
plt.legend()
plt.show()

# Fit with a quadratic polynomial
coefficients = np.polyfit(years, ratios, 2)
polynomial = np.poly1d(coefficients)
# Generate x values for the fitted curve
x_fit = np.linspace(years.min()-0.3, years.max()+1, 100)
y_fit = polynomial(x_fit)
# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.errorbar(years, ratios, yerr=uncertainties, fmt='o', ecolor='r', capsize=5, label='Student-Faculty Ratio')
plt.plot(x_fit, y_fit, 'b-', label='Quadratic Fit')
plt.title('Student-Faculty Ratio at CNU with Quadratic Fit')
plt.xlabel('Year')
plt.ylabel('Student-Faculty Ratio')
plt.grid(True)
plt.legend()
plt.show()

# Extract the uncertainties in the fitted values
y_fit_uncertainties = np.sqrt((2 * coefficients[0] * x_fit + coefficients[1])**2 * (uncertainties.mean()**2))
# Plot the fitted curve with uncertainties
plt.figure(figsize=(10, 6))
# plot the original data
plt.errorbar(years, ratios, yerr=uncertainties, fmt='o', ecolor='r', capsize=5, label='Student-Faculty Ratio')
plt.plot(x_fit, y_fit, 'b-', label='Trend Line')
plt.fill_between(x_fit, y_fit - y_fit_uncertainties, y_fit + y_fit_uncertainties, color='b', alpha=0.2, label='Trend Uncertainty')
plt.title('Student-Faculty Ratio at CNU with Trend Uncertainty')
plt.xlabel('Year')
plt.ylabel('Student-Faculty Ratio')
plt.grid(True)
plt.legend()
plt.show()
# Display the coefficients of the fitted polynomial
print("Fitted quadratic coefficients (a, b, c):", coefficients)
# where the polynomial is of the form: a*x^2 + b*x + c
# a = coefficients[0], b = coefficients[1], c = coefficients[2]

# Save the final plot to a file
plt.figure(figsize=(10, 6))
plt.errorbar(years, ratios, yerr=uncertainties, fmt='o', ecolor='r', capsize=5, label='Student-Faculty Ratio with Stat. Unc.')
plt.plot(x_fit, y_fit, 'b-', label='Trend Line')
plt.fill_between(x_fit, y_fit - y_fit_uncertainties, y_fit + y_fit_uncertainties, color='b', alpha=0.2, label='Trend Uncertainty')
plt.title('Student-Faculty Ratio at CNU with Trend Uncertainty')
plt.xlabel('Year')
plt.ylabel('Student-Faculty Ratio')
plt.grid(True)
plt.legend()
plt.savefig('student_faculty_ratio_trend.png')
plt.close()
# Save the fitted coefficients to a text fil