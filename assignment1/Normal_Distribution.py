import sys

# python Normal_Distribution.py AvgMaleHeight.txt DailyRainfall.txt WeeklyExpenditures.txt
samples = ["AvgMaleHeight.txt", "DailyRainfall.txt", "WeeklyExpenditures.txt"]


def format_data(file):
    with open(file, "r") as f:
        return [float(x) for x in f.read().splitlines()]


def compute_params(data):
    n = len(data)
    mean = sum(data) / n
    var = sum([(x - mean) ** 2 for x in data]) / (n)
    std = var ** 0.5
    return mean, var, std


if __name__ == "__main__":
    for file in sys.argv[1:] if len(sys.argv) > 1 else samples:
        data = format_data(file)
        mean, var, std = compute_params(data)

        print(f"\nfor {file}")
        print("mean: ", mean)
        print("var: ", var)
        print("std: ", std)
