# read in data from the file a1c.csv ... two columns - date (M/D/YYYY) and a1c value (float)
import pandas as pd

data = pd.read_csv('a1c.csv')
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
print(data)

# calculate a running average of the a1c values with a window of 4
data['a1c_avg'] = data['a1c'].rolling(window=4).mean()

import matplotlib.pyplot as plt
plt.plot(data['date'], data['a1c'], marker='o', linestyle='-', color='blue', label='A1C Value')
plt.plot(data['date'], data['a1c_avg'], marker='x', linestyle='--', color='red', label='Yearly Running Average')
plt.legend()
plt.xlabel('Date')
plt.ylabel('A1C Value')
plt.title('Hannah Murphy A1C Values Over Time')
plt.grid()
# set y axis limits to 5 to 10
plt.ylim(6, 9)
plt.show()