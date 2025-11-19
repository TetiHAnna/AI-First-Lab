import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

def load_and_clean_all_data(filenames):
    """
    Завантажує, об'єднує та очищує дані з усіх файлів маршрутів.
    """
    print("Завантаження та об'єднання даних з 10 файлів...")
    all_data_frames = []
    
    for i, filename in enumerate(filenames):
        try:
            data = pd.read_csv(filename, sep=r'\s+', header=None, 
                               names=['id', 'y', 'x', 'total_cps', 'value'], on_bad_lines='skip')
            data['route_id'] = i
            all_data_frames.append(data)
        except FileNotFoundError:
            print(f"ПОПЕРЕДЖЕННЯ: Файл '{filename}' не знайдено і буде пропущено.")

    if not all_data_frames:
        print("ПОМИЛКА: Не вдалося завантажити жодного файлу.")
        return None

    combined_data = pd.concat(all_data_frames, ignore_index=True)
    for col in ['x', 'y', 'value']:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    
    initial_rows = len(combined_data)
    combined_data.dropna(subset=['x', 'y', 'value'], inplace=True)
    combined_data = combined_data[(combined_data['x'] != 0) & (combined_data['y'] != 0)]
    rows_removed = initial_rows - len(combined_data)

    if rows_removed > 0:
        print(f"Очищення даних завершено. Видалено {rows_removed} некоректних рядків.")
    
    print(f"Фінальна кількість точок для аналізу: {len(combined_data)}.")
    return combined_data

def simulate_attack(dataset, num_anomalies=5):
    """
    Симулює кібератаку шляхом внесення аномальних значень.
    """
    attacked_data = dataset.copy()
    max_value = attacked_data['value'].max()
    anomaly_value = max_value * 5
    
    anomaly_indices = np.random.choice(attacked_data.index, num_anomalies, replace=False)
    
    attacked_data.loc[anomaly_indices, 'value'] += anomaly_value
    
    print(f"Симуляція атаки: внесено {num_anomalies} аномальні точки.")
    return attacked_data, anomaly_indices

def detect_anomalies(gpr_model, dataset):
    """
    Використовує навчену модель РГП для виявлення аномалій.
    """
    X_test = dataset[['x', 'y']].values
    y_test = dataset['value'].values
    
    print("Прогнозування та виявлення аномалій...")
    y_pred, y_std = gpr_model.predict(X_test, return_std=True)
    
    residuals = np.abs(y_test - y_pred)
    threshold = np.mean(y_std) * 6 
    
    detected_indices = dataset.index[residuals > threshold]
    
    print(f"Виявлено {len(detected_indices)} підозрілих точок.")
    return detected_indices

def plot_final_results(full_data, attacked_data, true_anomaly_indices, detected_anomaly_indices, gpr_model):
    """
    Візуалізує результати, включаючи карту невизначеності.
    """
    plot_data = full_data[full_data['route_id'] != 9]
    attacked_plot_data = attacked_data[attacked_data['route_id'] != 9]

    x_min, x_max = plot_data['x'].min(), plot_data['x'].max()
    y_min, y_max = plot_data['y'].min(), plot_data['y'].max()
    data_ratio = (y_max - y_min) / (x_max - x_min) if (x_max - x_min) != 0 else 1
    
    fig_width = 24
    fig_height = (fig_width / 3) * data_ratio + 2

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    fig.suptitle('Застосування РГП для побудови карти та виявлення атак', fontsize=18)

    vmin = plot_data['value'].min()
    vmax = plot_data['value'].max()

    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.ticklabel_format(style='plain', useOffset=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('Довгота', fontsize=12)
    axes[0].set_ylabel('Широта', fontsize=12)

    # --- 1. Інтерпольована карта (Чисті дані) ---
    ax1 = axes[0]
    grid_z_clean, _ = gpr_model.predict(np.c_[grid_x.ravel(), grid_y.ravel()], return_std=True)
    grid_z_clean = grid_z_clean.reshape(grid_x.shape)
    cf1 = ax1.contourf(grid_x, grid_y, grid_z_clean, cmap='viridis', levels=100, vmin=vmin, vmax=vmax)
    ax1.contour(grid_x, grid_y, grid_z_clean, colors='white', alpha=0.4, levels=10)
    ax1.scatter(plot_data['x'], plot_data['y'], s=5, c='black', alpha=0.5)
    ax1.set_title('1. Інтерпольована карта забруднення (РГП)', fontsize=14)
    
    # --- 2. Спотворена карта (Після атаки) ---
    ax2 = axes[1]
    points_attacked = attacked_plot_data[['x', 'y']].values
    values_attacked = attacked_plot_data['value'].values
    grid_z_attacked = griddata(points_attacked, values_attacked, (grid_x, grid_y), method='cubic')
    ax2.contourf(grid_x, grid_y, grid_z_attacked, cmap='viridis', levels=100, vmin=vmin, vmax=vmax*1.5) # Increase vmax to show spikes
    ax2.contour(grid_x, grid_y, grid_z_attacked, colors='white', alpha=0.4, levels=10)
    ax2.scatter(attacked_data.loc[true_anomaly_indices, 'x'], attacked_data.loc[true_anomaly_indices, 'y'],
                s=200, facecolors='none', edgecolors='yellow', linewidth=3, label='Симульована атака')
    ax2.set_title('2. Спотворена карта після атаки', fontsize=14)
    ax2.legend()

    # --- 3. Карта невизначеності та виявлені аномалії ---
    ax3 = axes[2]
    _, grid_std = gpr_model.predict(np.c_[grid_x.ravel(), grid_y.ravel()], return_std=True)
    grid_std = grid_std.reshape(grid_x.shape)
    cf3 = ax3.contourf(grid_x, grid_y, grid_std, cmap='magma', levels=100)
    fig.colorbar(cf3, ax=ax3, label='Рівень невизначеності (стандартне відхилення)')
    ax3.scatter(attacked_data.loc[detected_anomaly_indices, 'x'], attacked_data.loc[detected_anomaly_indices, 'y'],
                s=200, facecolors='none', edgecolors='cyan', linewidth=3, label='Виявлена аномалія')
    ax3.set_title('3. Карта невизначеності та виявлені аномалії', fontsize=14)
    ax3.legend()
    
    # Спільна кольорова шкала для перших двох карт
    fig.subplots_adjust(right=0.92, wspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5]) 
    fig.colorbar(cf1, cax=cbar_ax, label='Активність Cs-137')
    
    plt.show()

# --- Основне виконання скрипта ---
if __name__ == '__main__':
    all_files = [
        "Gals12080312.csv", "Gals12270312.csv", "Gals12380312.csv",
        "Gals13050312.csv", "Gals13220312.csv", "Gals13370312.csv",
        "Gals14070312.csv", "Gals14350312.csv", "Gals15130312.csv",
        "Gals15230312.csv"
    ]
    
    full_dataset = load_and_clean_all_data(all_files)
    
    if full_dataset is not None:
        print("Навчання моделі РГП на повних даних...")
        print("(Це може зайняти деякий час...)")
        
        X_train = full_dataset[['x', 'y']].values
        y_train = full_dataset['value'].values
        
        kernel = 1.0 * RBF(length_scale=0.001, length_scale_bounds=(1e-5, 1e-1)) + WhiteKernel(noise_level=1)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42, normalize_y=True)
        
        gpr.fit(X_train, y_train)
        print(f"Навчання завершено. Параметри ядра: {gpr.kernel_}")

        attacked_dataset, true_anomaly_indices = simulate_attack(full_dataset.copy())
        detected_anomaly_indices = detect_anomalies(gpr, attacked_dataset)
        
        plot_final_results(full_dataset, attacked_dataset, true_anomaly_indices, detected_anomaly_indices, gpr)

