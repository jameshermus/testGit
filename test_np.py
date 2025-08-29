import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create a 100x100 matrix of random values (standard normal distribution)
    matrix = np.random.randn(100, 100)

    # Flatten to 1D for histogram
    data = matrix.ravel()

    # Plot histogram (matplotlib for visualization)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, edgecolor="black")
    plt.title("Histogram of 100x100 Random Matrix Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

