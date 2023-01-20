# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:40:11 2023

@author: drink
"""

import os, shutil, sys
import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import gzip
#####
SAMPLE_SIZE = 100
new_names, new_model_names = [], []
sample_sessions = []
PATH_C3D_START = r"Y:\departments\research_and_development\sports_science\01_mocap_operations\pitching\v6\03_athletes"
PATH_C3D_COPY = r"C:\Users\kylew\Documents\GitHub\openbiomechanics\baseball_pitching\data\c3d"
PATH_META_SAVE = r"C:\Users\kylew\Documents\GitHub\openbiomechanics\baseball_pitching\data"
PATH_POI_SAVE = r"C:\Users\kylew\Documents\GitHub\openbiomechanics\baseball_pitching\data\poi"
PATH_FULL_SIG_SAVE = r"C:\Users\kylew\Documents\GitHub\openbiomechanics\baseball_pitching\data\full_sig"
SAVE_META_CSV = True
# ###################### Connect to DB ######################
# os.enviro 表示用戶的環境變量
db_user_ = os.environ['CLUSTER_USERNAME_DB_BIOMECH']
db_pw_ = os.environ['CLUSTER_PASSWORD_DB_BIOMECH']
db_dbname_ = os.environ['DATABASE_BIOMECH_PITCHING_DB']
db_host_ = os.environ['CLUSTER_HOST_DB_BIOMECH']
db_port_ = os.environ['CLUSTER_PORT_DB_BIOMECH']
# 初始化資料庫連接，使用pymysql模組。(若是要用其他的只要更換pymysql即可)
engine_string = 'mysql+mysqlconnector://'+db_user_ +':'+db_pw_+'@'+db_host_+':'+str(db_port_)+'/'+db_dbname_
# 建立連線引擎
# create_engine用來連接sql的url，

engine = create_engine(engine_string,connect_args={'auth_plugin': 'mysql_native_password'}, echo=False)
# engine用connect進行連線，
#Get Data from Biomech DB
cnx = mysql.connector.connect(host=db_host_,
                                         database=db_dbname_,
                                         user=db_user_,
                                         password=db_pw_,
                                         port =db_port_,
                                         auth_plugin='mysql_native_password')
#藉由execute執行查詢語句。

# To create a cursor, use the cursor() method of a connection object:
cursor = cnx.cursor()

query = 'select distinct session_pitch from bp_force_plate where rear_force_x is not null;'
#讀取sql data
d = pd.read_sql(query, cnx)
force_plate_pitches = tuple(pd.unique(d.session_pitch))

query = "select u.user, s.session, poi.session_pitch, s.session_mass_kg, s.session_height_m, poi.pitch_speed_mph from bp_sessions s left join bp_poi_metrics poi using(session) left join bp_users u using(user) where (s.irb=1) and (poi.pitch_type='FF') and (s.playing_level not in ('mlb', 'other')) and (poi.session_pitch in {}) and (u.user not in (552));"
d = pd.read_sql(query.format(force_plate_pitches), cnx)
users = pd.unique(d.user).tolist()

sample_users = np.random.choice(users, SAMPLE_SIZE, replace=False)

for user in sample_users:
    user_sessions = d[d.user==user].session.tolist()
    # pick one session
    session = np.random.choice(user_sessions, 1)[0]
    sample_sessions.append(session)
sample_sessions = tuple(sample_sessions)

query = "select s.*, poi.session_pitch, poi.pitch_speed_mph, u.date_of_birth from bp_sessions s left join bp_poi_metrics poi using(session) left join bp_users u using(user) where (s.session in {}) and (poi.pitch_type='FF');"
d1 = pd.read_sql(query.format(sample_sessions), cnx)
d1['session_date'] = pd.to_datetime(d1['session_date'])
d1['date_of_birth'] = pd.to_datetime(d1['date_of_birth'])
d1['age_yrs'] = d1['session_date'] - d1['date_of_birth']
d1['age_yrs'] = np.round(d1['age_yrs'].apply(lambda x: x.days/365.25),2)
d1 = d1[['user', 'session', 'session_pitch', 'session_mass_kg', 'session_height_m', 'age_yrs', 'playing_level', 'pitch_speed_mph']]

query = "select u.user, s.session, pos.session_pitch, pos.filename FROM raw_bp_positions pos left join bp_poi_metrics poi using(session_pitch) left join bp_sessions s on(s.SESSION=pos.session) left join bp_users u ON (s.user=u.user) where (s.session in {}) and (poi.pitch_type='FF') and u.name not in ('Luisa Gauci');"
d2 = pd.read_sql(query.format(sample_sessions), cnx)
d2['user'] = d2['user'].astype(str).apply(lambda x: x.zfill(6))
d2['session'] = d2['session'].astype(str).apply(lambda x: x.zfill(6))
modelnames = d2['filename'].apply(lambda x: '_'.join(x.split('_')[0:3]) + '_model.c3d')
d2['modelnames'] = modelnames

for x in d2.iterrows():
    parts = x[1]['filename'].split('_')
    base_new = '_'.join(parts[3:])
    base_new = x[1]['user'] + '_' + x[1]['session'] + '_' + base_new
    base_new = base_new.replace('.json', '.c3d')
    new_names.append(base_new)
    model_new = x[1]['user'] + '_' + x[1]['session'] + '_model.c3d'
    new_model_names.append(model_new)

d2['filename_new'] = new_names
d2['modelname_new'] = new_model_names

# join d1 and d2 on session_pitch
d = d1.merge(d2, on=['session_pitch'], how='left')
d.drop(columns=['session_y', 'user_y'], inplace=True)
d.rename(columns={'session_x': 'session', 'user_x': 'user'}, inplace=True)
d.filename = d.filename.apply(lambda x: x.replace('.json', '.c3d'))
sample_session_pitches = tuple(d.session_pitch.tolist())

if SAVE_META_CSV:
    d.drop(columns=['filename', 'modelnames']).to_csv(os.path.join(PATH_META_SAVE, "metadata.csv"), index=False)
    
for root, dirs, files in os.walk(PATH_C3D_START):
    for file in files:
        if file.endswith('.c3d'):
            if file in d.filename.tolist():
                filename_new = d[d.filename==file].filename_new.tolist()[0]
                start_path = os.path.join(root, file)
                folder_athlete = filename_new.split('_')[0]
                end_path = os.path.join(PATH_C3D_COPY, folder_athlete, filename_new)
                if not os.path.exists(os.path.join(PATH_C3D_COPY, folder_athlete)):
                    os.makedirs(os.path.join(PATH_C3D_COPY, folder_athlete))
                shutil.copyfile(start_path, end_path)
                print('Copied {} to {}'.format(file, end_path))
            elif file in d.modelnames.tolist():
                filename_new = d[d.modelnames==file].modelname_new.tolist()[0]
                start_path = os.path.join(root, file)
                folder_athlete = filename_new.split('_')[0]
                end_path = os.path.join(PATH_C3D_COPY, folder_athlete, filename_new)
                if not os.path.exists(os.path.join(PATH_C3D_COPY, folder_athlete)):
                    os.makedirs(os.path.join(PATH_C3D_COPY, folder_athlete))
                shutil.copyfile(start_path, end_path)
                print('Copied {} to {}'.format(file, end_path))
            else:
                pass
            
d = pd.read_csv(os.path.join(PATH_META_SAVE, 'metadata.csv'))
sample_session_pitches = tuple(d.session_pitch.tolist())

# save force plate full signal data from session_pitches
query = "select fp.session_pitch, fp.time, fp.rear_force_x, fp.rear_force_y, fp.rear_force_z, fp.lead_force_x, fp.lead_force_y, fp.lead_force_z, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time from bp_force_plate fp left join bp_events e using(session_pitch) where session_pitch in {};".format(sample_session_pitches)
force_plate_data = pd.read_sql(query, cnx)
force_plate_data.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'force_plate.csv'), index=False)

query = "select ef.*, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time from bp_energy_flow ef left join bp_events e using(session_pitch) where session_pitch in {};".format(sample_session_pitches)
energy_flow_data = pd.read_sql(query, cnx)
energy_flow_data.drop(columns=['session_pitch_time'], inplace=True)
energy_flow_data.drop(columns=[x for x in energy_flow_data.columns if 'total' in x], inplace=True)
energy_flow_data.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'energy_flow.csv'), index=False)

query = "SELECT jm.*, jf.*, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time FROM bp_joint_forces jf LEFT JOIN bp_joint_moments jm USING(session_pitch_time) LEFT JOIN bp_events e ON(jm.session_pitch=e.session_pitch) WHERE jm.session_pitch in {};".format(sample_session_pitches)
forces_moments = pd.read_sql(query, cnx)
forces_moments.drop(columns=['session_pitch_time'], inplace=True)
forces_moments.iloc[:, 50:52] = np.nan
forces_moments.dropna(axis=1, how='all', inplace=True)
forces_moments.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'forces_moments.csv'), index=False)

query = "select ja.*, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time from bp_joint_angles ja left join bp_events e using(session_pitch) where session_pitch in {};".format(sample_session_pitches)
joint_angles = pd.read_sql(query, cnx)
joint_angles.drop(columns=['session_pitch_time'], inplace=True)
joint_angles.drop(columns=[x for x in joint_angles.columns if 'virtual_lab' in x], inplace=True)
joint_angles.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'joint_angles.csv'), index=False)

query = "select jv.*, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time from bp_joint_velos jv left join bp_events e using(session_pitch) where session_pitch in {};".format(sample_session_pitches)
joint_velos = pd.read_sql(query, cnx)
joint_velos.drop(columns=['session_pitch_time'], inplace=True)
joint_velos.drop(columns=[x for x in joint_velos.columns if 'sig_mag' in x], inplace=True)
joint_velos.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'joint_velos.csv'), index=False)

query = "select lm.*, e.pkh_time, e.fp_force_plates_time_10perc as fp_10_time, e.fp_force_plates_time_100perc as fp_100_time, e.MER_time, e.BR_time, e.MIR_time from bp_landmarks lm left join bp_events e using(session_pitch) where session_pitch in {};".format(sample_session_pitches)
landmarks = pd.read_sql(query, cnx)
landmarks.drop(columns=['session_pitch_time'], inplace=True)
landmarks.drop(columns=[x for x in landmarks.columns if '_velo_' in x], inplace=True)
landmarks.to_csv(os.path.join(PATH_FULL_SIG_SAVE, 'landmarks.csv'), index=False)

query = "select * from bp_poi_metrics where session_pitch in {};".format(sample_session_pitches)
poi_metrics = pd.read_sql(query, cnx)
poi_metrics.drop(columns=['lead_knee_extension_angular_velo_max_legacy'], inplace=True)
poi_metrics.to_csv(os.path.join(PATH_POI_SAVE, 'poi_metrics.csv'), index=False)