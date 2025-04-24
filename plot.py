import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
file_path = "execution_times.csv"  
df = pd.read_csv(file_path)

# Pivot the data for better structure
sequential = df[df['Type'] == 'SEQUENTIAL']
parallel = df[df['Type'] == 'PARALLEL']

# Ensure the data aligns by Resolution and Channels
merged = sequential.merge(
    parallel,
    on=['Resolution', 'Channels'],
    suffixes=('_sequential', '_parallel')
)

# Calculate speedup
merged['Speedup'] = merged['Time_sequential'] / merged['Time_parallel']

# Plotting
fig, ax = plt.subplots()

for channels in merged['Channels'].unique():
    subset = merged[merged['Channels'] == channels]
    ax.plot(
        subset['Resolution'], 
        subset['Speedup'], 
        marker='o', 
        label=f'{channels} Channel(s)'
    )

ax.set_title('Speedup vs Resolution')
ax.set_xlabel('Resolution')
ax.set_ylabel('Speedup')
ax.legend()
ax.grid(True)

plt.show()
