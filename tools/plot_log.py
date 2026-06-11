#!/usr/bin/env python3
"""
AGV Log Plotter — đọc CSV từ agv_logger.py và plot
Usage:
    python plot_log.py agv_log_20260610_120000.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────
# Load data
# ─────────────────────────────────────────
def load(filename):
    df = pd.read_csv(filename)
    df["time_s"] = (df["timestamp_us"] - df["timestamp_us"].iloc[0]) / 1e6
    return df

# ─────────────────────────────────────────
# Plot
# ─────────────────────────────────────────
def plot(df, filename):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"AGV Log — {filename}", fontsize=12)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    t = df["time_s"]

    # ── 1. RPM 4 bánh ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df["fl_rpm"], label="FL")
    ax1.plot(t, df["fr_rpm"], label="FR")
    ax1.plot(t, df["rl_rpm"], label="RL")
    ax1.plot(t, df["rr_rpm"], label="RR")
    ax1.set_title("Encoder RPM — 4 bánh")
    ax1.set_ylabel("RPM")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. PWM command ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["pwm_fl"], label="FL")
    ax2.plot(t, df["pwm_fr"], label="FR")
    ax2.plot(t, df["pwm_rl"], label="RL")
    ax2.plot(t, df["pwm_rr"], label="RR")
    ax2.set_title("PWM Command (-1000 ~ +1000)")
    ax2.set_ylabel("PWM")
    ax2.set_ylim(-1100, 1100)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── 3. Velocity actual vs setpoint ─────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, df["vx"],    label="vx actual",  linewidth=1.5)
    ax3.plot(t, df["sp_vx"], label="vx setpoint", linestyle="--", alpha=0.7)
    ax3.plot(t, df["vy"],    label="vy actual",  linewidth=1.5)
    ax3.plot(t, df["sp_vy"], label="vy setpoint", linestyle="--", alpha=0.7)
    ax3.set_title("Velocity Vx, Vy — actual vs setpoint")
    ax3.set_ylabel("m/s")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. Yaw rate actual vs setpoint ─────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, df["wz"],    label="wz actual",  linewidth=1.5)
    ax4.plot(t, df["sp_wz"], label="wz setpoint", linestyle="--", alpha=0.7)
    ax4.set_title("Yaw Rate Wz — actual vs setpoint")
    ax4.set_ylabel("rad/s")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. IMU — gyro z (heading rate) ─────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t, df["gz"],    label="gz (gyro)",  color="purple")
    ax5.plot(t, df["yaw"],   label="yaw (deg)",  color="orange", linestyle="--")
    ax5.set_title("IMU — Gyro Z & Yaw")
    ax5.set_ylabel("rad/s  /  deg")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6. IMU — accel XY ──────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t, df["ax"], label="ax", color="red")
    ax6.plot(t, df["ay"], label="ay", color="green")
    ax6.plot(t, df["az"], label="az", color="blue", alpha=0.5)
    ax6.set_title("IMU — Accelerometer (m/s²)")
    ax6.set_ylabel("m/s²")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, alpha=0.3)

    # X label cho row cuối
    for ax in [ax5, ax6]:
        ax.set_xlabel("Time (s)")

    # ── Drift summary ───────────────────────
    try:
        drift_yaw = df["yaw"].iloc[-1] - df["yaw"].iloc[0]
        duration  = t.iloc[-1]
        rpm_std   = {
            "FL": df["fl_rpm"].std(),
            "FR": df["fr_rpm"].std(),
            "RL": df["rl_rpm"].std(),
            "RR": df["rr_rpm"].std(),
        }
        print(f"\n[SUMMARY] Duration: {duration:.1f}s")
        print(f"[SUMMARY] Yaw drift: {drift_yaw:.2f} deg")
        print(f"[SUMMARY] RPM std: FL={rpm_std['FL']:.1f}  FR={rpm_std['FR']:.1f}  "
              f"RL={rpm_std['RL']:.1f}  RR={rpm_std['RR']:.1f}")
    except Exception:
        pass

    plt.savefig(filename.replace(".csv", "_plot.png"), dpi=150, bbox_inches="tight")
    print(f"[DONE] Saved: {filename.replace('.csv', '_plot.png')}")
    plt.show()

# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_log.py <agv_log_file.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"[INFO] Loading: {filename}")
    df = load(filename)
    print(f"[INFO] {len(df)} rows, {df['time_s'].iloc[-1]:.1f}s duration")
    plot(df, filename)
