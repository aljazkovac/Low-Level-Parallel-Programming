#import matplotlib.pyplot as plt

labels = []
scenarioData = []
hugeScenarioData = []
scenarioBoxData = []

with open("results2.txt", 'r') as f:
    
    lines = f.readlines()

    for _ in range(4):
        labels.append(lines.pop(0).strip())

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        scenarioData.append((val1+val2)/2)

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        hugeScenarioData.append((val1+val2)/2)

        lines.pop(0)
        val1 = int(lines.pop(0).strip())
        val2 = int(lines.pop(0).strip())
        scenarioBoxData.append((val1+val2)/2)

print(labels)
print(scenarioData)
print(hugeScenarioData)
print(scenarioBoxData)
