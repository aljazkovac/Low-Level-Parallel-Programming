import matplotlib.pyplot as plt
import numpy as np

# Initial values from earlier measurments
labels = ["SEQ (A1)"]
scenarioData = [4339]
hugeScenarioData = [29829]
scenarioBoxData = [72]

with open("results2.txt", 'r') as f:
    
    lines = f.readlines()

    for _ in range(4):
        labels.append(lines.pop(0).strip().split(" ")[0])

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        scenarioData.append(round((val1+val2)/2))

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        hugeScenarioData.append(round((val1+val2)/2))

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        scenarioBoxData.append(round((val1+val2)/2))

print(labels)
print(scenarioData)
print(hugeScenarioData)
print(scenarioBoxData)

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
r1 = ax.bar(x - width, scenarioData, width, label='scenario.xml')
r2 = ax.bar(x, hugeScenarioData, width, label='hugeScenario.xml')
r3 = ax.bar(x + width, scenarioBoxData, width, label='scenario_box.xml')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Execution time (milliseconds)')
ax.set_title('Execution times for different implementations')
ax.set_xticks(x, labels)
ax.legend()

# ax.bar_label(r1, padding=3)
# ax.bar_label(r2, padding=3)
# ax.bar_label(r3, padding=3)

fig.tight_layout()

# plt.show()
plt.savefig("execution_times.jpg")