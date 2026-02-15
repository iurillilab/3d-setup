import re
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.internals.blocks import external_values
import ssm
from tqdm import tqdm

FNAME_REGEX = re.compile(
    r"m(\d+)_s(\d+)_(cricket|object)\.xlsx$", re.IGNORECASE
)

def parse_filename(fname: str):
    m = FNAME_REGEX.match(fname)
    if not m:
        return None
    mouse_id = int(m.group(1))
    cond = m.group(3).lower()
    return mouse_id, cond

def session_index(name: str) -> int:
        """
        Extracts the integer after '_s' in filenames like 'm003_s007_cricket.xlsx'.
        Returns it as an int (e.g. 7).
        """
        m = re.search(r'_s(\d+)', name)
        if m is None:
            # fallback if pattern not found; put these at the end
            return 999999
        return int(m.group(1))

def preprocess_session_df(
        df: pd.DataFrame,
        feature_cols_linear,
        angle_cols,
        fps:float=60.0,
        target_dt:float = 0.1
        ):
    df = df.copy()
    # 1) angle -> sin and cos assuming them in degress
    for col in angle_cols:
        if col not in df.columns:
            continue
        theta_rad = np.deg2rad(df[col].to_numpy())
        df[col+"_sin"] = np.sin(theta_rad)
        df[col+"_cos"] = np.cos(theta_rad)

    # drop original cols
    df = df.drop(columns = [c for c in angle_cols if c in df.columns])
    angle_sin_cos_cols = [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]
    feature_cols = [c for c in feature_cols_linear if c in df.columns] + angle_sin_cos_cols

    df_feat = df[feature_cols]

    df_smooth = df_feat.rolling(window=3, center=True).mean().dropna()

    # we now downsample to approx the target time scale:
    downsample_factor = max(1, int(round(target_dt * fps)))
    df_proc = df_smooth.iloc[::downsample_factor]
    X = df_proc.to_numpy()
    return X, feature_cols
def build_sequences(
    paths,
    feature_cols_linear,
    angle_cols,
    fps=60.0,
    target_dt=0.1,
    mouse_filter=None,
    condition_filter=None,
):
    sequences = []
    metas = []
    all_cols_ref = None

    for path in paths:
        if path.name == "startTimes.xlsx":
            continue

        # ---- NEW: safely parse filename ----
        parsed = parse_filename(path.name)
        if parsed is None:
            print(f"Skipping file with unrecognized name: {path.name}")
            continue
        mouse, condition = parsed
        # ------------------------------------

        sid = session_index(path.name)

        if mouse_filter is not None and mouse != mouse_filter:
            continue
        if condition_filter is not None and condition != condition_filter:
            continue

        df = pd.read_excel(path)

        X, cols = preprocess_session_df(
            df,
            feature_cols_linear=feature_cols_linear,
            angle_cols=angle_cols,
            fps=fps,
            target_dt=target_dt,
        )

        if X.shape[0] < 20:
            # too short, skip
            continue

        if all_cols_ref is None:
            all_cols_ref = cols
        else:
            assert cols == all_cols_ref, "Feature mismatch across sessions"

        sequences.append(X)
        metas.append({
            "mouse": mouse,
            "condition": condition,
            "session": sid,
            "file": path.name,
        })

    return sequences, metas, all_cols_ref

def zscore_sequences(sequences):
    all_data = np.concatenate(sequences, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std[std==0] = 1.0

    zseqs = [(seq-mean) / std for seq in sequences]
    return zseqs, mean, std

def debug_session(zseq, z_states, meta, feat_cols, dt=0.1, out_path=None):
    print("=== Debug session ===")
    print("len(z_states):", len(z_states))
    print("zseq shape:", zseq.shape)
    print("any NaN in zseq:", np.isnan(zseq).any())
    print("states unique:", np.unique(z_states))
    print("meta:", meta)
    print("=====================")

    if len(z_states) == 0 or zseq.shape[0] == 0:
        print("Empty sequence or states → nothing to plot.")
        return

    # pick feature to plot
    if "dist_head" in feat_cols:
        feat_idx = feat_cols.index("dist_head")
        feat = zseq[:, feat_idx]
        feat_label = "dist_head (z-scored)"
    else:
        feat_idx = 0
        feat = zseq[:, feat_idx]
        feat_label = feat_cols[0] + " (z-scored)"

    t = np.arange(len(z_states)) * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # feature trace
    axes[0].plot(t, feat[:len(t)])
    axes[0].set_ylabel(feat_label)
    axes[0].set_title(
        f"Mouse {meta['mouse']} | {meta['condition']} | session {meta['session']}"
    )

    # ethogram (states over time)
    im = axes[1].imshow(
        z_states[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=[t[0], t[-1], 0, 1],
    )
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("State")

    fig.colorbar(im, ax=axes[1], label="State")

    plt.tight_layout()

    if out_path is not None:
        print(f"Saving figure to {out_path}")
        fig.savefig(out_path, dpi=150)
    else:
        plt.show()

def dwell_times(z):
    if len(z) == 0:
        return np.array([])
    durs = []
    cur = z[0]
    run = 1
    for s in z[1:]:
        if s == cur:
            run += 1
        else:
            durs.append(run)
            cur = s
            run = 1
    durs.append(run)
    return np.array(durs)

def make_state_feature_df(
        zseqs,
        state_seqs,
        metas,
        feat_cols,
        mean_vec,
        std_vec,
        save_path:Path=None
        ):
    rows = []

    for seq_z, z, meta in zip(zseqs, state_seqs, metas):
        seq_real = seq_z * std_vec + mean_vec
        T = seq_real.shape[0]

        #find angle
        try:
            i_sin = feat_cols.index("facing_angle_sin")
            i_cos = feat_cols.index("facing_angle_cos")
            have_angle = True
        except:
            have_angle=False

        for t in range(T):
            row = {
                    "state": int(z[t]),
                    "mouse": meta["mouse"],
                    "condition": meta["condition"],
                    "session": meta["session"],
                }
            for j, name in enumerate(feat_cols):
                row[name] = seq_real[t, j]

                if have_angle:
                    s = seq_real[t, i_sin]
                    c= seq_real[t, i_cos]
                    angle_rad = np.arctan2(s, c)
                    angle_deg = np.degrees(angle_rad)
                    row["facing_angle_deg"] = angle_deg
                rows.append(row)
        df = pd.DataFrame(rows)
        if save_path is not None:
            df.to_csv(save_path)
        return df

def sequences_to_dataframe(sequences, metas, feat_cols, fps=60.0):
    """Flatten list of sequences into a single dataframe with metadata attached."""

    all_rows = []
    for seq, meta in zip(sequences, metas):
        T = seq.shape[0]
        df_seq = pd.DataFrame(seq, columns=feat_cols)

        # time values
        df_seq["time"] = np.arange(T) / fps

        # attach metadata
        for k, v in meta.items():
            df_seq[k] = v

        all_rows.append(df_seq)

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all

def add_zscored_columns(df, mean_vec, std_vec, feat_cols):
    Z = (df[feat_cols].values - mean_vec) / std_vec
    Zdf = pd.DataFrame(Z, columns=[f"{c}_z" for c in feat_cols])
    df = pd.concat([df.reset_index(drop=True), Zdf], axis=1)
    return df
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dir_features", required=True,)
    parser.add_argument("--dt", type=float, default=0.15, help="Target time scale (dt) in seconds")

    args = parser.parse_args()
    features_dir = Path(args.dir_features)
    feature_paths = sorted(features_dir.glob("*.xlsx"))
    feature_paths = [Path(path) for path in feature_paths]


    FEATURE_COLS_LINEAR = [
        "dist_head",
        "dist",
        "dist_change",
        "rel_bearing",
        "height",
        "height_scaled",
        "head_acc",
        "rigidbody_acc",
        "speed_head_fwd",
        "head_speed",
        "trunk_speed",
        "radial_vel",
        "tangential_vel",
        "nose_tail_distance",
        "forepawL_tail_distance",
        "forepawR_tail_distance",
        "forepaw_target_distance",
        "angle_head_body_speed",
        "dist_head_scaled",
        "dist_scaled",
        "dist_change_scaled",
        "omega_theta_trunk",
    ]
    #
    # angular features (in degrees) -> we'll convert to sin/cos
    ANGLE_COLS = [
        "facing_angle",
        "rel_angle_target",
        "angle_head_body_axis",
        "angle_head_body_l",
        "angle_head_body_r",
        "ori_allBody",
        "ori_trunk",
        "ori_head",
    ]
    #

    sequences, metas, feat_cols = build_sequences(
            feature_paths,
            feature_cols_linear=FEATURE_COLS_LINEAR,
            angle_cols=ANGLE_COLS,
            fps=60.0,
            target_dt=args.dt,)

    print(f"Created sequences for model\n {len(sequences)=} \n {feat_cols=}")

    df_all = sequences_to_dataframe(sequences, metas, feat_cols, fps=60.0)

    out_path = f"/users/thomasbush/Downloads/features_all_dt{args.dt:.3f}.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Saved raw features dataframe to {out_path}")

    # print("Start model fitting")

    # K=6
    # D = zseqs[0].shape[1]
    # lags = 1 # AR(1)

    # hmm = ssm.HMM(
    #         K=K,
    #         D=D,
    #         transitions="sticky",
    #         transition_kwargs=dict(kappa=50.0),
    #         observations="gaussian",)
    #         # observation_kwargs=dict(lags=lags)
    #         # )

    # lls =hmm.fit(
    #         zseqs,
    #         method="em",
    #         num_iters=50,
    #         init_method="kmeans",
    #         verbose=True)
    # print("Model fitted")

    # state_seq = [hmm.most_likely_states(seq) for seq in zseqs]

    # idx = 0

    # debug_session(
    #         zseq=zseqs[idx],
    #         z_states=state_seq[idx],
    #         meta = metas[idx],
    #         feat_cols=feat_cols,
    #         dt=0.15,
    #         out_path= "/users/thomasbush/Downloads/ethogram_session0.png",
    #         )
    # all_durs = np.concatenate([dwell_times(z) for z in state_seq])
    # print("Median dwell length (steps):", np.median(all_durs))
    # print("Median dwell length (seconds):", np.median(all_durs) * 0.1)

    # print("Transition matrix:\n", hmm.transitions.transition_matrix)
    # print("Diagonal (self-transition probs):", np.diag(hmm.transitions.transition_matrix))

    # state_means = np.zeros((K, D))
    # state_counts = np.zeros(K)

    # for seq, z in zip(zseqs, state_seq):
    #     for k in range(K):
    #         mask = (z == k)
    #         if mask.any():
    #             state_means[k] += seq[mask].mean(axis=0)
    #             state_counts[k] += 1

    # state_means /= np.maximum(state_counts[:, None], 1)

    # print("Per-state means (z-scored):")
    # for k in range(K):
    #     print(f"\nState {k}:")
    #     for name, val in zip(feat_cols, state_means[k]):
    #         print(f"  {name:20s} {val:6.3f}")
    # df_states = make_state_feature_df(
    #         zseqs,
    #         state_seq,
    #         metas,
    #         feat_cols,
    #         mean_vec,
    #         std_vec,
    #         Path(f"/users/thomasbush/Downloads/features-states_all.csv"),)
    # print(df_states.head())

