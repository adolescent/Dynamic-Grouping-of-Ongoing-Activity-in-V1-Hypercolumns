# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:36:55 2022

@author: ZR
"""
import pandas as pd
from tqdm import tqdm


def Get_Frame_Response(series,cell_tuning_dic,thres = 2,clip = 5,OD_thres = 0.5):
    '''
    This function will calculate OD,Orientation response in all series.

    Parameters
    ----------
    series : (pd Frame)
        Pre-processed series train, can be generated by Preprocessor
    cell_tuning_dic : (Dic)
        Cell Tuning Dic, also generated by caiman process.
    thres : (int), optional
        Thres hold to define spike. The default is 2.
    clip : (int), optional
        Clip up extrem high value. The default is 5.
    OD_thres : (float), optional
        Threshold to determine an OD response. The default is 5.
        

    Returns
    -------
    frame_response : (pd Frame)
        Pandas frame of Single-Frame response.

    '''
    # clip firing frame, get response here.
    firing_data = series[series>thres].fillna(0).clip(upper = clip)
    # get tuning
    actune = pd.DataFrame(columns = ['OD','Orien'])
    acn = firing_data.index
    for i,cc in enumerate(acn):
        tc = cell_tuning_dic[cc]
        if tc['Fitted_Orien'] != 'No_Tuning':
            actune.loc[cc] = [tc['OD']['Tuning_Index'],tc['Fitted_Orien']]
    actune['Orien_Group'] = (((actune['Orien']+22.5)%180)//45)*45 # Given Orien+-22.5
    all_frame_name = firing_data.columns
    all_cell_name_t = actune.index
    all_cell_Num = len(all_cell_name_t)
    LE_cell_Num = (actune['OD']>OD_thres).sum()
    RE_cell_Num = (actune['OD']<-OD_thres).sum()
    Orien0_cell_Num = (actune['Orien_Group']==0).sum()
    Orien45_cell_Num = (actune['Orien_Group']==45).sum()
    Orien90_cell_Num = (actune['Orien_Group']==90).sum()
    Orien135_cell_Num = (actune['Orien_Group']==135).sum()
    tuned_data = firing_data.loc[all_cell_name_t]
    # generate frame response
    spike_info_column = ['All_Num','All_spike','All_prop','LE_Num','LE_spike','LE_prop','RE_Num','RE_spike','RE_prop','Orien0_Num','Orien0_spike','Orien0_prop','Orien45_Num','Orien45_spike','Orien45_prop','Orien90_Num','Orien90_spike','Orien90_prop','Orien135_Num','Orien135_spike','Orien135_prop']
    frame_response = pd.DataFrame(0,columns = spike_info_column,index = all_frame_name)
    for c_frame_name in tqdm(all_frame_name):
        c_frame = tuned_data[c_frame_name]
        firing_cells = c_frame[c_frame>0]
        frame_response.loc[c_frame_name,'All_Num'] = len(firing_cells)
        frame_response.loc[c_frame_name,'All_spike'] = firing_cells.sum()
        fire_cell_names = list(firing_cells.index)
        # then cycle cells to count tuning response.
        for cc in fire_cell_names:
            c_tune = actune.loc[cc,:]
            if c_tune['OD']>OD_thres: # LE cell
                frame_response.loc[c_frame_name,'LE_Num'] +=1
                frame_response.loc[c_frame_name,'LE_spike'] += firing_cells[cc]
            elif c_tune['OD']<-OD_thres: # RE cell
                frame_response.loc[c_frame_name,'RE_Num'] +=1
                frame_response.loc[c_frame_name,'RE_spike'] += firing_cells[cc]
            # orien counter
            if c_tune['Orien_Group'] == 0 : # Orien 0 cell
                frame_response.loc[c_frame_name,'Orien0_Num'] +=1
                frame_response.loc[c_frame_name,'Orien0_spike'] += firing_cells[cc]
            elif c_tune['Orien_Group'] == 45 : # Orien 45 cell
                frame_response.loc[c_frame_name,'Orien45_Num'] +=1
                frame_response.loc[c_frame_name,'Orien45_spike'] += firing_cells[cc]
            elif c_tune['Orien_Group'] == 90 : # Orien 90 cell
                frame_response.loc[c_frame_name,'Orien90_Num'] +=1
                frame_response.loc[c_frame_name,'Orien90_spike'] += firing_cells[cc]
            elif c_tune['Orien_Group'] == 135 : # Orien 135 cell
                frame_response.loc[c_frame_name,'Orien135_Num'] +=1
                frame_response.loc[c_frame_name,'Orien135_spike'] += firing_cells[cc]
        # at last, calculate response propotion
        frame_response['All_prop'] = frame_response['All_spike']/all_cell_Num
        frame_response['LE_prop'] = frame_response['LE_spike']/LE_cell_Num
        frame_response['RE_prop'] = frame_response['RE_spike']/RE_cell_Num
        frame_response['Orien0_prop'] = frame_response['Orien0_spike']/Orien0_cell_Num
        frame_response['Orien45_prop'] = frame_response['Orien45_spike']/Orien45_cell_Num
        frame_response['Orien90_prop'] = frame_response['Orien90_spike']/Orien90_cell_Num
        frame_response['Orien135_prop'] = frame_response['Orien135_spike']/Orien135_cell_Num
        #define cell_num_dic
        cell_num_dic = {}
        cell_num_dic['LE'] = LE_cell_Num
        cell_num_dic['RE'] = RE_cell_Num
        cell_num_dic['Orien0'] = Orien0_cell_Num
        cell_num_dic['Orien45'] = Orien45_cell_Num
        cell_num_dic['Orien90'] = Orien90_cell_Num
        cell_num_dic['Orien135'] = Orien135_cell_Num
        
    
    return frame_response,cell_num_dic,actune