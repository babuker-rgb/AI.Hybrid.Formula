def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples, 8))
    y = np.zeros((n_samples, 3))

    x_min = -np.log(1 - D_MIN)
    x_max = -np.log(1 - D_MAX)

    for i in range(n_samples):
        # مكونات خام
        api_raw = np.random.uniform(85, 95)
        binder_raw = np.random.uniform(0.5, 5.0)
        mgst_raw = np.random.uniform(0.01, 1.2)
        pvpp_raw = np.random.uniform(0.5, 6.0)
        mcc_raw = np.random.uniform(0.0, MCC_MAX)
        pressure = np.random.uniform(80, PRESSURE_MAX)
        speed = np.random.uniform(1, 50)
        granule = np.random.uniform(30, 250)

        # تطبيع المكونات
        api, binder, pvpp, mgst, mcc = normalize_components(api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw)
        X[i] = [api, mcc, pvpp, mgst, binder, pressure, speed, granule]

        # حساب الكثافة باستخدام معادلة Heckel
        k = np.random.uniform(0.01, 0.05)
        A = np.random.uniform(0.5, 2.5)
        x_new = k * pressure + A
        D_target = 1 - np.exp(-x_new)
        D_target = np.clip(D_target, D_MIN, D_MAX)

        # تقليل الضوضاء العشوائية
        noise_d = np.random.normal(0, 0.005)   # أقل من النسخة السابقة
        D = np.clip(D_target + noise_d, D_MIN, D_MAX)

        # حساب الصلابة (Tensile Strength)
        sigma0 = 5.0
        b = 2.5
        porosity = 1.0 - D
        tensile_base = sigma0 * np.exp(-b * porosity)

        # تأثير المكونات
        api_effect = 1.0 - 0.003 * (api - 90)
        binder_effect = 1.0 + 0.02 * (binder - 2.5)
        mgst_effect = 1.0 - 0.05 * (mgst - 0.2)
        pvpp_effect = 1.0 - 0.01 * (pvpp - 3.0)
        speed_effect = 1.0 - 0.001 * (speed - 10)

        strength = tensile_base * api_effect * binder_effect * mgst_effect * pvpp_effect * speed_effect
        strength = np.clip(strength + np.random.normal(0, 0.02), 0.5, 6.0)

        # حساب المرونة (Elastic Recovery)
        er_base = 1.8 + 0.25 * (api - 85)/10 + 0.05 * (speed - 10)/30 - 0.08 * (pressure - 100)/150
        er = np.clip(er_base * (1.0 - 0.1 * (D - 0.4)) + np.random.normal(0, 0.02), 0.5, 4.0)

        y[i] = [D, strength, er]

    feature_names = ['API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
                     'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm']
    df = pd.DataFrame(X, columns=feature_names)
    df['Density'] = y[:, 0]
    df['Tensile_Strength_MPa'] = y[:, 1]
    df['Elastic_Recovery_%'] = y[:, 2]
    return df, feature_names
