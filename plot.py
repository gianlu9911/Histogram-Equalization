import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
file_path = "execution_times.csv"  # Replace with the correct file name if needed
df = pd.read_csv(file_path)

# Pivot the data for better structure
sequential = df[df['type'] == 'SEQUENTIAL']
parallel = df[df['type'] == 'PARALLEL']

# Ensure the data aligns by Resolution and Channels
merged = sequential.merge(
    parallel,
    on=['Resolution', 'channels'],
    suffixes=('_sequential', '_parallel')
)

# Calculate speedup
merged['Speedup'] = merged['time_sequential'] / merged['time_parallel']

# Plotting
fig, ax = plt.subplots()

for channels in merged['channels'].unique():
    subset = merged[merged['channels'] == channels]
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
