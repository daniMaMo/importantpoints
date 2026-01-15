import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def EDist(ys, xs, Adjx, Adjy):
    return (
        np.sqrt((Adjx[:, 1] - xs)**2 + (Adjy[:, 1] - ys)**2) +
        np.sqrt((Adjx[:, 0] - xs)**2 + (Adjy[:, 0] - ys)**2)
    )

def PDist(ys, xs, Adjx, Adjy):
    slopes = (Adjy[:, 1] - Adjy[:, 0]) / (Adjx[:, 1] - Adjx[:, 0])
    constants = Adjy[:, 1] - slopes * Adjx[:, 1]
    return np.abs(slopes * xs - ys + constants) / np.sqrt(slopes**2 + 1)

def VDist(ys, xs, Adjx, Adjy):
    slopes = (Adjy[:, 1] - Adjy[:, 0]) / (Adjx[:, 1] - Adjx[:, 0])
    constants = Adjy[:, 1] - slopes * Adjx[:, 1]
    yhat = slopes * xs + constants
    return np.abs(yhat - ys)

def PIPs(ys, n_PIPs, type_of_dist=1, pflag=0):
    """
        Identifies Perceptually Important Points (PIPs) of a price series.

        Parameters
        ----------
        ys : array-like
            Price series (1D array).
        n_PIPs : int
            Number of requested PIPs minus 1.
        type_of_dist : int
            1 = Euclidean Distance (ED)
            2 = Perpendicular Distance (PD)
            3 = Vertical Distance (VD)
        pflag : int
            If 1, plot the PIPs.

        Returns
        -------
        PIPxy : ndarray (n_PIPs + 1, 2)
            Columns: [x-coordinate, y-coordinate]
        """

    ys = np.asarray(ys).flatten()
    l = len(ys)
    xs = np.arange(1, l + 1)

    # Binary indexation of PIPs
    PIP_points = np.zeros(l)
    PIP_points[0] = 1  # first observation
    PIP_points[-1] = 1  # last observation

    currentstate = 2  # initial PIPs

    while currentstate <= n_PIPs:

        Existed_Pips = np.where(PIP_points == 1)[0]
        Existed_Pips = Existed_Pips + 1
        currentstate = len(Existed_Pips)

        locator = np.zeros((l, currentstate))
        locator[:] = np.nan

        # --- Compute distances to existing PIPs (x-distance only)
        for j in range(currentstate):
            locator[:, j] = np.abs(xs - Existed_Pips[j])

        # --- Find two closest adjacent PIPs for each point
        b1 = np.zeros(l, dtype=int)
        b2 = np.zeros(l, dtype=int)
        Adjacent = np.zeros((l, 2), dtype=int)

        for i in range(l):
            if np.all(np.isnan(locator[i])):
                continue

            b1[i] = np.nanargmin(locator[i])
            locator[i, b1[i]] = np.nan
            b2[i] = np.nanargmin(locator[i])

            Adjacent[i, 0] = Existed_Pips[b1[i]]
            Adjacent[i, 1] = Existed_Pips[b2[i]]

        # --- Build adjacency matrices
        Adjx = Adjacent.astype(float)
        Adjy = np.column_stack((ys[Adjacent[:, 0] - 1],
                                ys[Adjacent[:, 1] - 1]))

        # Existing PIPs are not candidates
        Adjx[Existed_Pips - 1, :] = np.nan
        Adjy[Existed_Pips - 1, :] = np.nan

        # --- Distance calculation
        if type_of_dist == 1:
            D = EDist(ys, xs, Adjx, Adjy)
        elif type_of_dist == 2:
            D = PDist(ys, xs, Adjx, Adjy)
        else:
            D = VDist(ys, xs, Adjx, Adjy)

        Dmax = np.nanargmax(D)
        PIP_points[Dmax] = 1
        currentstate += 1

    Existed_Pips = np.where(PIP_points == 1)[0] + 1
    PIPxy = np.column_stack((Existed_Pips, ys[Existed_Pips - 1]))

    # --- Plot
    if pflag == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, label='Price series')
        plt.plot(PIPxy[:, 0], PIPxy[:, 1], 'r*', label='PIPs')
        plt.legend()
        plt.show()

    return PIPxy

#------------Examples----------------------------
DATA_FOLDER = "data_arrays"

n_PIPs = 9          # Control parameter: the algorithm returns n_PIPs + 1 points
type_of_dist = 2    # 1=ED, 2=PD, 3=VD
pflag = 1           # If 1, plot the PIPs

for file_name in os.listdir(DATA_FOLDER):

    # Build full path to the CSV file
    file_path = os.path.join(DATA_FOLDER, file_name)

    # --- Read CSV file
    df = pd.read_csv(file_path)

    # --- Convert 'Date' column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Filter data for year 2021
    df_2021 = df[df["Date"].dt.year == 2021]

    # Skip assets without data for the selected year
    if df_2021.empty:
        print(f"{file_name}: no data for 2021")
        continue

    # --- Extract Adjusted Close prices
    ys = df_2021["Adj Close"].values

    # --- Run PIPs algorithm
    print(f"\nActivo: {file_name}")
    PIPxy = PIPs(
        ys=ys,
        n_PIPs=n_PIPs,
        type_of_dist=type_of_dist,
        pflag=pflag
    )

    print("PIPs (x, y):")
    print(PIPxy)
