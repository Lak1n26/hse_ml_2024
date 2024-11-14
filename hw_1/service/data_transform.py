import re
import pandas as pd
import numpy as np



def get_correct_mileage(val):
    if val == 'nan':
        return None
    val = val.lower().split(' ')
    if val[1] == 'kmpl':        
        return float(val[0])
    return float(val[0]) * 1.4


def get_clear_torque(val):
    if 'nm' in val.lower() and 'kgm' in val.lower():
        return float(re.search(r'\d+(\.\d+)?', val.lower()).group())
    elif 'nm' in val.lower():  # ньютон-метр
        val = ''.join(val.split(' '))
        if 'nm@rpm' in val.lower():
            return float(re.search(r'\d+(\.\d+)?', val.lower()).group())
        return float(re.search(r'\d+(\.\d+)?', val.lower()).group())
    elif 'kgm' in val.lower():  # килограмм-метр: 1kgm = 9.80665nm
        val = ''.join(val.split(' '))
        if 'kgm@rpm' in val.lower():
            return float(re.search(r'\d+(\.\d+)?', val.lower()).group()) * 9.80665
        return float(re.search(r'\d+(\.\d+)?', val.lower()).group()) * 9.80665
    else:
        return float(re.search(r'\d+(\.\d+)?', val.lower()).group())

def get_max_torque_rpm(val):
    new_val = float(re.search(r'\d+(\.\d+)?(?!.*\d)', val).group())
    if new_val < 6:
        return new_val * 1000
    return new_val


def get_final_data(df, scaler):
    df['name'] = df['name'].str.split(' ').str[0]
    df['mileage'] = df['mileage'].astype(str).apply(get_correct_mileage)
    df['engine'] = df['engine'].str.split(' ').str[0].astype(float)
    df['max_power'] = df['max_power'].str.split(' ').str[0].apply(lambda x: '0' if x == '' else x).astype(float)
    df['max_torque_rpm'] = df['torque'].apply(get_max_torque_rpm)
    df['torque'] = df['torque'].apply(get_clear_torque)
    df['horses'] = df['max_power'] / (df['engine'] / 1000)
    df['year_square'] = df['year'] ** 2
    df['engine_square'] = df['engine'] ** 2
    df['max_power_square'] = df['max_power'] ** 2
    df['km_driven_sqrt'] = df['km_driven'] ** 0.5
    df['more_than_1_owner'] = np.where(df.owner.isin(['Second Owner','Third Owner', 'Fourth & Above Owner']), 1, 0)
    df['more_than_2_owners'] = np.where(df.owner.isin(['Third Owner', 'Fourth & Above Owner']), 1, 0)
    df['km_per_age'] = df['km_driven'] / (2021 - df['year'])

    template = pd.DataFrame([0]*53).T
    template.columns = ['seller_type_Individual', 'fuel_Diesel', 'name_Datsun',
       'owner_Third Owner', 'name_Jaguar', 'name_Chevrolet',
       'name_Mercedes-Benz', 'owner_Test Drive Car', 'mileage', 'name_BMW',
       'seats', 'name_Ambassador', 'km_driven', 'name_Nissan',
       'transmission_Manual', 'fuel_CNG', 'fuel_LPG', 'name_Skoda',
       'name_Audi', 'name_Toyota', 'transmission_Automatic', 'engine',
       'torque', 'seller_type_Dealer', 'km_driven_sqrt',
       'seller_type_Trustmark Dealer', 'name_Tata', 'name_Volvo',
       'owner_First Owner', 'max_power', 'name_Hyundai', 'year_square',
       'name_Mahindra', 'more_than_1_owner', 'name_Volkswagen', 'name_Lexus',
       'name_Jeep', 'engine_square', 'more_than_2_owners', 'max_power_square',
       'name_Maruti', 'max_torque_rpm', 'name_Ford', 'name_Renault',
       'owner_Second Owner', 'km_per_age', 'name_Honda',
       'owner_Fourth & Above Owner', 'fuel_Petrol', 'name_Mitsubishi',
       'horses', 'name_Fiat', 'year']
    
    # исходя из обучающей выборки заполняем значения
    # если числовое значение - берем его
    # если категориальное - 1/0
    for col in df.columns:
        if col in template.columns:
            template[col] = df[col].iloc[0]
        elif col+'_'+df[col].iloc[0] in template.columns:
            template[col+'_'+df[col].iloc[0]] = 1

    df = template.copy()
    
    # стандартизируем
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque',
       'max_torque_rpm', 'horses', 'year_square', 'more_than_2_owners',
       'more_than_1_owner', 'engine_square', 'max_power_square',
       'km_driven_sqrt', 'km_per_age']
    cat_cols = list(set(df.columns) - set(num_cols))
    
    df = pd.concat([pd.DataFrame(scaler.transform(df[num_cols]), columns=df[num_cols].columns), df[cat_cols]], axis=1)
    return df[template.columns]
