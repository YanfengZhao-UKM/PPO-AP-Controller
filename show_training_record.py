import numpy as np
from matplotlib import pyplot as plt

def render(measure_metrics_records):
    # Generate data from the measure_metrics_records
    x = np.arange(int(len(measure_metrics_records) / 4)) + 1
    TAR_S = measure_metrics_records[0::4, 1].flatten() * 100
    TAR = measure_metrics_records[0::4, 2].flatten() * 100
    TIR = measure_metrics_records[0::4, 3].flatten() * 100
    TBR = measure_metrics_records[0::4, 4].flatten() * 100
    TBR_S = measure_metrics_records[0::4, 5].flatten() * 100
    TAR_S_mean = round(np.mean(TAR_S), 2)
    TAR_mean = round(np.mean(TAR), 2)
    TIR_mean = round(np.mean(TIR), 2)
    TBR_mean = round(np.mean(TBR), 2)
    TBR_S_mean = round(np.mean(TBR_S), 2)
    TAR_S_max = measure_metrics_records[2::4, 1].flatten() * 100
    TAR_max = measure_metrics_records[2::4, 2].flatten() * 100
    TIR_max = measure_metrics_records[2::4, 3].flatten() * 100
    TBR_max = measure_metrics_records[2::4, 4].flatten() * 100
    TBR_S_max = measure_metrics_records[2::4, 5].flatten() * 100
    TAR_S_min = measure_metrics_records[2::4, 1].flatten() * 100
    TAR_min = measure_metrics_records[3::4, 2].flatten() * 100
    TIR_min = measure_metrics_records[3::4, 3].flatten() * 100
    TBR_min = measure_metrics_records[3::4, 4].flatten() * 100
    TBR_S_min = measure_metrics_records[3::4, 5].flatten() * 100

    Reward = measure_metrics_records[0::4, 0].flatten()
    Reward_mean = round(np.mean(Reward), 2)
    Reward_max = measure_metrics_records[2::4, 0].flatten()
    Reward_min = measure_metrics_records[3::4, 0].flatten()
    LBGI = measure_metrics_records[0::4, 6].flatten()
    HBGI = measure_metrics_records[0::4, 7].flatten()
    RISK = measure_metrics_records[0::4, 8].flatten()
    LBGI_mean = round(np.mean(LBGI), 2)
    HBGI_mean = round(np.mean(HBGI), 2)
    RISK_mean = round(np.mean(RISK), 2)
    LBGI_max = measure_metrics_records[2::4, 6].flatten()
    HBGI_max = measure_metrics_records[2::4, 7].flatten()
    RISK_max = measure_metrics_records[2::4, 8].flatten()
    LBGI_min = measure_metrics_records[3::4, 6].flatten()
    HBGI_min = measure_metrics_records[3::4, 7].flatten()
    RISK_min = measure_metrics_records[3::4, 8].flatten()

    # Create the time of the glucose range axis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))

    # 左侧子图占50%
    ax1.set_position([0.1, 0.15, 0.4, 0.7])

    # 右侧上下两个子图各占25%
    ax2.set_position([0.6, 0.55, 0.35, 0.3])
    ax3.set_position([0.6, 0.15, 0.35, 0.3])

    # Plot the time of the glucose range lines
    ax1.plot(x, TAR_S, label='Severe Hyper:'+ str(TAR_S_mean)+'%')
    ax1.fill_between(x, TAR_max, TAR_S_min, alpha=0.2)

    ax1.plot(x, TAR, label='Hyper:'+ str(TAR_mean)+'%')
    ax1.fill_between(x, TAR_max, TAR_min, alpha=0.2)

    ax1.plot(x, TIR, label='Normal:'+str(TIR_mean)+'%')
    ax1.fill_between(x, TIR_max, TIR_min, alpha=0.2)

    ax1.plot(x, TBR, label='Hypo:'+str(TBR_mean)+'%')
    ax1.fill_between(x, TBR_max, TBR_min, alpha=0.2)

    ax1.plot(x, TBR_S, label='Severe Hypo:'+str(TBR_S_mean)+'%')
    ax1.fill_between(x, TBR_S_max, TBR_S_min, alpha=0.2)

    ax1.set_title('Glucose range time percent')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Percent')
    ax1.legend()

    # Create the reward and risk axes
    #fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8))

    ax2.plot(x, Reward, label='Reward:'+ str(Reward_mean))
    ax2.fill_between(x, Reward_max, Reward_min, alpha=0.2)

    ax2.set_title('Reward Plot')
    #ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()

    #fig, ax3 = plt.subplots(figsize=(4, 8))

    ax3.plot(x, HBGI, label='HBGI:'+ str(HBGI_mean))
    ax3.fill_between(x, HBGI_max, HBGI_min, alpha=0.2)

    ax3.plot(x, LBGI, label='LBGI:'+ str(LBGI_mean))
    ax3.fill_between(x, LBGI_max, LBGI_min, alpha=0.2)

    ax3.plot(x, RISK, label='RISK:' + str(RISK_mean))
    ax3.fill_between(x, RISK_max, RISK_min, alpha=0.2)

    ax3.set_title('RISK Plot')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('RISK')
    ax3.legend()

    fig.suptitle('adolescent#001',y=0.95)
    plt.show()


file_path = 'training_model/adolescent#001_Beta_seed1_T1DSimHistoryEnv/mesure_metrics.npy'

loaded_data = np.load(file_path)

render(loaded_data[2:])






