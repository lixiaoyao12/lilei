"""
    # File:'kddcup99date',
    # Author:'li lei',
    # Date:2024/5/29
    # Description:''
"""
import numpy as np
from sklearn.datasets import fetch_kddcup99


def save_KddCup99_tcpdump_tcp_tofile(fpath):
    """
    Fetch (from sklearn.datasets) the KddCup99 tcpdump dataset, extract tcp samples, and save to local as csv
    :param fpath: local file path to save the dataset
    """
    # Fetch the data
    kddcup99 = fetch_kddcup99(subset='10percent', return_X_y=True)
    X, y = kddcup99.data, kddcup99.target

    # Filter for TCP samples (this is a placeholder, you'll need to implement the actual filtering)
    # tcp_indices = ...
    # X_tcp = X[tcp_indices]
    # y_tcp = y[tcp_indices]

    # For now, we'll just save the entire dataset
    header = 'duration, src_bytes, dst_bytes, land, urgent, hot, #failed_login, '
    'logged_in, #compromised, root_shell, su_attempted, #root, #file_creations, #shells, '
    '#access_files, is_guest_login, count, srv_cnt, serror_rate, srv_serror_rate, rerror_rate, '
    'srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_cnt,'
    'dst_host_srv_cnt, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rt,'
    'dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, '
    'dst_host_rerror_rate, dst_host_srv_rerror_rate, label'  # Your full header here
    np.savetxt(fpath, np.column_stack((X, y)), delimiter=',', fmt='%.6f', header=header)


# Example usage:
save_KddCup99_tcpdump_tcp_tofile('./data/kddcup99_data.csv')