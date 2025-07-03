#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import MLP as MLP
import compute_invariants as ci
import os
import csv


def append_to_csv_line(csv_path, row_num, value):
    """
    Appends `value` to the end of line `row_num` (1-based) in a CSV.
    If the file doesn’t exist, it’s created. If row_num>1, earlier rows
    are created as empty.
    """
    rows = []
    if os.path.exists(csv_path):
        # Read existing rows
        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))
    # Ensure we have at least row_num rows
    while len(rows) < row_num:
        rows.append([])
    # Append the value to the desired row
    rows[row_num-1].append(str(value))
    # Write back (creates file if missing)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def replace_columns_with_row_average_single(csv_path, cols, output_path=None):
    """
    Reads a CSV, collapses specified columns (1-based indices) into one column 
    containing the row-wise average, and writes back (or returns) the modified DataFrame.
    
    Parameters:
        csv_path (str): Path to the input CSV.
        cols (list of int): 1-based column indices to average and replace.
        output_path (str, optional): If provided, saves modified CSV to this path.
        
    Returns:
        pd.DataFrame: The modified DataFrame with one averaged column.
    """
    df = pd.read_csv(csv_path, header=None)
    # Convert to 0-based indices
    zero_based = sorted([c - 1 for c in cols])
    # Compute row-wise mean across specified columns
    row_mean = df.iloc[:, zero_based].mean(axis=1)
    # Drop the original specified columns
    df_dropped = df.drop(columns=zero_based)
    # Insert the new average column at the position of the first specified column
    insert_pos = zero_based[0]
    df_dropped.insert(loc=insert_pos, column=f'Avg_{cols[0]}_{cols[-1]}', value=row_mean)
    # Optionally save to CSV
    if output_path:
        df_dropped.to_csv(output_path, header=False, index=False)
    return df_dropped

def plot_csv(
    file_path,
    labels=None,
    smooth_window=10  # how many points to average over
):
    """
    Reads a CSV and plots column 1 vs column 0 as a single smooth line.
      – X-axis comes from the 2nd CSV column
      – Y-axis comes from the 1st CSV column
    """
    # -- load and smooth --
    df = pd.read_csv(file_path, header=None)
    df_smooth = df.rolling(window=smooth_window, axis=0, min_periods=1).mean()

    # pull x from column 1, y from column 0
    x = df_smooth.iloc[:, 1]
    y = df_smooth.iloc[:, 0]

    # -- styling --
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # single-line plot
    label = (labels[0] if labels and len(labels)>0 else None)
    ax.plot(
        x,
        y,
        linewidth=2.5,
        alpha=0.9,
        label=label
    )

    # -- axes labels & title (unchanged) --
    ax.set_xlabel('Regularization Weight λ', fontsize=14)
    ax.set_ylabel('Test RMSE',           fontsize=14)
    ax.set_title('Effect of λ on Test RMSE after 100 training Epochs (averaged over 5 seeds)', fontsize=16)

    # -- grid: only horizontal lines --
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(False,   axis='x')

    # -- legend outside, semi-transparent --
    if label is not None:
        leg = ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True
        )
        leg.get_frame().set_alpha(0.5)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot each column of a CSV as a smooth line against epochs.'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='new4.csv',
        help='Path to your CSV file (default: new.csv)'
    )
    parser.add_argument(
        '--labels',
        nargs='*',
        default=[ 'Training Set: 360', 'Training Set: 866', 'Training Set: 240', 'Training Set: 120'
        ],
        help='Custom labels for each column (in order). Defaults to 12 noise levels.'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=10,
        help='Window size for rolling‐average smoothing (default: 10)'
    )
    args = parser.parse_args()

    #15 epochs #20 seeds 

    #40 epochs #10 seeds

    #100 epochs #5 seeds

    


    MLP.EPOCHS=100
    open('new4.csv', 'w').close()

    #246 Datasamples with 0.1 noise 6 shapes 
    MLP.FILE2='360_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()

    for lam in range(60):
        MLP.LAMBDA = lam/35.0
        results = []
        for seed in range(5):
            MLP.set_seed(seed)
            res = MLP.main()
            results.append(res)
        avg = sum(results) / len(results)
        append_to_csv_line("new4.csv", lam+1, avg)
        append_to_csv_line("new4.csv",lam+1,lam/35.0)

        


    

    plot_csv(args.csv_file, smooth_window=1)

     