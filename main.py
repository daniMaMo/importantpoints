import numpy as np
import matplotlib.pyplot as plt

def RW(ys, w, pflag=0):
    """
        Identifies regional peaks and bottoms of a price series using
        a rolling window (RW) of size 2w + 1.

        Parameters
        ----------
        ys : array-like
            Price series (1D array or column vector).
        w : int
            Half window size. Total window size is 2w + 1.
        pflag : int, optional
            If 1, generates a plot. Default is 0.

        Returns
        -------
        Peaks : ndarray of shape (n, 2)
            Columns: [y-coordinate, x-coordinate] of peaks.
        Bottoms : ndarray of shape (k, 2)
            Columns: [y-coordinate, x-coordinate] of bottoms.
        """
    # Ensure ys is a 1D NumPy array
    ys = np.asarray(ys).flatten()
    l = len(ys)

    # Preallocation (equivalent to zeros(l,2) in MATLAB)
    Peaks_Bottoms = np.zeros((l, 2))

    # MATLAB: for i = w+1 : l-w
    # Python (0-based): i = w : l-w-1
    for i in range(w, l - w):
        # Check peak
        if ys[i] > np.max(ys[i - w:i]) and ys[i] > np.max(ys[i + 1:i + w + 1]):
            Peaks_Bottoms[i, 0] = 1

        # Check bottom
        if ys[i] < np.min(ys[i - w:i]) and ys[i] < np.min(ys[i + 1:i + w + 1]):
            Peaks_Bottoms[i, 1] = 1

    # Indices of peaks and bottoms
    P_Indx = np.where(Peaks_Bottoms[:, 0] == 1)[0]
    B_Indx = np.where(Peaks_Bottoms[:, 1] == 1)[0]

    # MATLAB: Peaks = [ys(P_Indx), P_Indx]
    # MATLAB indices start at 1 → add +1 if you want exact equivalence
    Peaks = np.column_stack((ys[P_Indx], P_Indx + 1))
    Bottoms = np.column_stack((ys[B_Indx], B_Indx + 1))

    # Plot if requested
    if pflag == 1:
        plt.figure()
        plt.plot(ys, label="Price series")
        plt.plot(Peaks[:, 1] - 1, Peaks[:, 0], 'ro', label="Peaks")
        plt.plot(Bottoms[:, 1] - 1, Bottoms[:, 0], 'r*', label="Bottoms")
        plt.legend()
        plt.show()

    return Peaks, Bottoms

ys = np.array([1, 3, 2, 5, 1, 4, 2, 6, 1])
w = 1

Peaks, Bottoms = RW(ys, w, pflag=1)

print("Peaks:")
print(Peaks)

print("\nBottoms:")
print(Bottoms)
