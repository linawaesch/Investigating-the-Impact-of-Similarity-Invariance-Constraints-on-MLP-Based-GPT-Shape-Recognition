#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#import MLP_percentages as MLP
import MLP as MLP
import compute_invariants as ci


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
    Reads a CSV and plots each column as a separate smooth line.
    X-axis: row number (epochs), Y-axis: value (Test RMSE).
    """
    # -- load and optionally smooth --
    df = pd.read_csv(file_path, header=None)
    df_smooth = df.rolling(window=smooth_window, axis=0, min_periods=1).mean()
    epochs = df.index + 1

    n_cols = df_smooth.shape[1]
    cmap = plt.get_cmap('tab20', n_cols)  # up to 20 distinct colors

    # -- styling --
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # -- plot each column --
    for idx, col in enumerate(df_smooth.columns):
        label = (labels[idx] if labels and idx < len(labels)
                 else f'Col {idx+1}')
        ax.plot(
            epochs,
            df_smooth[col],
            linewidth=2.5,
            alpha=0.9,
            color=cmap(idx),
            label=label
        )

    # -- axes labels & title --
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Test RMSE', fontsize=14)
    ax.set_title('Training Set Size vs. Test RMSE', fontsize=16)

    # -- grid: only horizontal lines --
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(False, axis='x')

    # -- legend outside, semi-transparent --
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
        default='new2.csv',
        help='Path to your CSV file (default: new.csv)'
    )
    parser.add_argument(
        '--labels',
        nargs='*',
        default=[
            'Training Set: 24', 'Training Set: 60', 'Training Set: 120', 'Training Set: 240', 'Training Set: 360', 'Training Set: 720'
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

    MLP.EPOCHS=1000
    open('new2.csv', 'w').close()


    MLP.FILE2='24_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()


    for i in range(3):
        MLP.set_seed(i*8)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[1,2,3], output_path='new2.csv')

    MLP.FILE2='60_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()

    for i in range(3):
        MLP.set_seed(i*2)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[2,3,4], output_path='new2.csv')



    MLP.FILE2='120_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()

    for i in range(3):
        MLP.set_seed(i*3)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[3,4,5], output_path='new2.csv')

    

    MLP.FILE2='240_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()
    for i in range(3):
        MLP.set_seed(i*3)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[4,5,6], output_path='new2.csv')

   


    MLP.FILE2='360_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()

    for i in range(3):
        MLP.set_seed(i*3)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[5,6,7], output_path='new2.csv')

    MLP.FILE2='720_010_data-generation.csv'
    ci.FILENAME=MLP.FILE2
    ci.main()

    for i in range(3):
        MLP.set_seed(i*3)
        MLP.main()
    replace_columns_with_row_average_single('new2.csv',cols=[6,7,8], output_path='new2.csv')
   

    plot_csv(args.csv_file, labels=args.labels, smooth_window=args.smooth)

    
