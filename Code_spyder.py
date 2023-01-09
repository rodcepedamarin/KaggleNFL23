#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1. Libraries and modules
###############################################################################
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from dateutil import parser
import pickle
from IPython.display import clear_output
import timeit
import itertools
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2. Data
###############################################################################
games = pd.read_csv('data/games.csv')
scout = pd.read_csv('data/pffScoutingData.csv')
players = pd.read_csv('data/players.csv')
plays = pd.read_csv('data/plays.csv')
week = {}
for i in range(1, 7):
    week[i] = pd.read_csv('data/week' + str(i) + '.csv')
    if i == 1:
        week_t = week[i].copy()
    else:
        week_t = pd.concat([week_t, week[i]])
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3. Functions
###############################################################################
def intocm(df, col_name):
    """
    Function tu turn feet-inch to cm
    Parameters
    ----------
    df : dataframe
    col_name : name of column that contains the measure in f-i
        DESCRIPTION.
    Returns
    -------
    df : datadrame
        new df modified.
    """
    _df = df[col_name].str.split("-", n = 1, expand = True)
    _df[0] = _df[0].apply(lambda x: float(x))
    _df[1] = _df[1].apply(lambda x: float(x))
    df['height'] = _df[0] * 30.48 + _df[1] * 2.54
    return df

def intersection_vectorized(a1, a2, b1, b2):
    def general_eq(p1, p2):
        #ax + by + c = 0
        #https://www.cuemath.com/geometry/intersection-of-two-lines/
        x1 = p1[:, 0]
        y1 = p1[:, 1]
        x2 = p2[:, 0]
        y2 = p2[:, 1]
        result = np.zeros([len(b1), 3])
        
        _ind = pd.Series(x1 == x2)
        result[_ind[_ind == True].index, 0] = x1[_ind[_ind == True].index]
        
        _ind = pd.Series(y1 == y2)
        result[_ind[_ind == True].index, 1] = y1[_ind[_ind == True].index]
        
        _ind = pd.Series((x1 != x2) * (y1 != y2))
        result[_ind[_ind == True].index, 1] = 1
        
        m = (y2[_ind[_ind == True].index] - y1[_ind[_ind == True].index]) / \
            (x2[_ind[_ind == True].index] - x1[_ind[_ind == True].index])
        b = y1[_ind[_ind == True].index] - m * x1[_ind[_ind == True].index]
        
        result[_ind[_ind == True].index, 0] = - m
        result[_ind[_ind == True].index, 2] = - b
        return result
    
    r = general_eq(a1, a2)
    r2 = general_eq(b1, b2)
    
    inter = np.transpose(np.array([(r[:, 1] * r2[:, 2] - r2[:, 1] * r[:, 2]) / (r[:, 0] * r2[:, 1] - r2[:, 0] * r[:, 1]),
                                   (r[:, 2] * r2[:, 0] - r2[:, 2] * r[:, 0]) / (r[:, 0] * r2[:, 1] - r2[:, 0] * r[:, 1])]))
    return inter


def perp_points_2(center, point, point_new, orient): # Que pasa si son abscisas o ordenadas!!!
    """
    Function that given two points of a line, it creates a perpendicular line to the first
    line that goes through another given point
    Parameters
    ----------
    center : np.array
        first point - location of QB..
    point : np.array
        second point of the line.
    point_new : np.array
        point that new perp line is passing through.

    Returns
    -------
    pto_ret:
        TYPE: np.array
        second point that characterizes a line (used with point_new).

    """
    orient = pd.Series(orient)
    pto_ret = center * 0 #Structure
    
    _ind = orient[(orient == 90) | (orient == 270)].index
    pto_ret[_ind, 0] = point_new[_ind, 0]
    pto_ret[_ind, 1] = point_new[_ind, 1] - 1
    
    _ind = orient[(orient == 0) | (orient == 360) | (orient == 180)].index
    pto_ret[_ind, 0] = point_new[_ind, 0] - 1
    pto_ret[_ind, 1] = point_new[_ind, 1]
    
    _ind = orient[(orient != 0) & (orient != 90) & (orient != 180) & (orient != 270) & (orient != 360)].index
    _slope = ((point[_ind, 1] - center[_ind, 1]) / (point[_ind, 0] - center[_ind, 0]))
    _slope_2 = -1 / _slope
    inter_2  =    point_new[_ind, 1] - _slope_2 * point_new[_ind, 0]
    
    pto_ret[_ind, 0] = 0
    pto_ret[_ind, 1] = inter_2
    
    return pto_ret

def two_points_point_angle_2(center, orient):
    """
    Function that generates two points of a line given one point and an
    angle (the angle stars at 12 hours and turns clockwise)
    Parameters
    ----------
    center : np.array
        first point - location of QB.
    orient : float
        angle orientation of QB when passing.

    Returns
    -------
    x1_y1 : np.array
        the second point in the line.
    """
    # Calculate x1_y1 to create a line using two points
    orient = pd.Series(orient)
    orient.loc[orient[orient < 0].index]  = orient.loc[orient[orient < 0].index] + 360
    orient.loc[orient[orient > 360].index]  = orient.loc[orient[orient > 360].index] - 360
    
    x1_y1 = center * 0
    
    _ind = orient[(orient > 0) & (orient < 90)].index
    x1_y1[_ind, 0] = center[_ind, 0] + 1
    x1_y1[_ind, 1] = center[_ind, 1] + np.tan(np.deg2rad(90 - orient[_ind]))
    
    _ind = orient[(orient > 90) & (orient < 180)].index
    x1_y1[_ind, 0] = center[_ind, 0] + 1
    x1_y1[_ind, 1] = center[_ind, 1] - np.tan(np.deg2rad(orient[_ind] - 90))
    
    _ind = orient[(orient > 180) & (orient < 270)].index
    x1_y1[_ind, 0] = center[_ind, 0] + 1
    x1_y1[_ind, 1] = center[_ind, 1] + np.tan(np.deg2rad(90 - orient[_ind] - 180))
    
    _ind = orient[(orient > 270) & (orient < 360)].index
    x1_y1[_ind, 0] = center[_ind, 0] + 1
    x1_y1[_ind, 1] = center[_ind, 1] - np.tan(np.deg2rad(orient[_ind] -180 - 90))
    
    _ind = orient[(orient == 0) | (orient == 180)].index
    x1_y1[_ind, 0] = center[_ind, 0]
    x1_y1[_ind, 1] = center[_ind, 1] + 1
    
    _ind = orient[(orient == 90) | (orient == 270)].index
    x1_y1[_ind, 0] = center[_ind, 0] + 1
    x1_y1[_ind, 1] = center[_ind, 1]

    return x1_y1


def two_points_point_angle(center, orient):
    """
    Function that generates two points of a line given one point and an
    angle (the angle stars at 12 hours and turns clockwise)
    Parameters
    ----------
    center : np.array
        first point - location of QB.
    orient : float
        angle orientation of QB when passing.

    Returns
    -------
    x1_y1 : np.array
        the second point in the line.
    """
    # Calculate x1_y1 to create a line using two points

    if orient < 0: orient = orient + 360
    elif orient > 360: orient = orient - 360
    
    if (orient > 0 and orient < 90):
        x1_y1 = np.array([center[0] + 1,
                          center[1] + np.tan(np.deg2rad(90 - orient))])
    elif (orient > 90 and orient < 180):
        x1_y1 = np.array([center[0] + 1,
                          center[1] - np.tan(np.deg2rad(orient - 90))])
    elif (orient > 180 and orient < 270):
        orient_2 = orient - 180
        x1_y1 = np.array([center[0] + 1,
                          center[1] + np.tan(np.deg2rad(90 - orient_2))])
    elif (orient > 270 and orient < 360):
        orient_2 = orient - 180
        x1_y1 = np.array([center[0] + 1,
                          center[1] - np.tan(np.deg2rad(orient_2 - 90))])
    elif (orient == 0 or orient == 180):
        x1_y1 = np.array([center[0], center[1] + 1])
    elif (orient == 90 or orient == 270):
        x1_y1 = np.array([center[0] + 1, center[1]])

    return x1_y1

def height_weight_to_width_new(height, weight):
    """
    Function that calculates width based on height and weight.
    The minimum width is 0.25 * height
    The maxumum width is 0.75 * height
    Parameters
    ----------
    height : float
    weight : float
    Returns: width_ratio
    -------
    """
    xs = [1, 1.8]
    ys = np.array([0.35, 0.55])
    
    ratio = weight / height
    try: ratio[ratio > 1.8] = 1.8
    except: pass
    try: ratio[ratio < 1] = 1
    except: pass
    width_ratio = np.interp(ratio, xs, ys)
    return width_ratio

def distance_vector(x0, x1, y0, y1):
    return np.sqrt((x0 - x1) ** 2 + \
        (y0 - y1) ** 2) * 100


def check_angle_reach_2(qb_position, radio, def_position, orient, width_angle):
    """
    Check if defender is within angle reach of the QB
    Parameters
    ----------
    qb_position : np.array
        X, y coordinates of QB.
    radio : integer
        radio of vision of QB.
    array : coordinate/s of defenders
        array.
    orient : float
        angle orientation of QB when passing.

    Returns
    -------
    TYPE: Bool, wheter it is inside or not of the QBs angle vision
    """
    try: # If def_position is vector
        x = def_position[:, 0]
        y = def_position[:, 1]
    except: # If def_position is single value
        x = def_position[0]
        y = def_position[1]
        
    # First the circle
    _circ =  (x - qb_position[:, 0])** 2 + (y - qb_position[:, 1]) ** 2 <= radio ** 2
    # Second the arc

    angle = np.transpose(np.array([orient + (width_angle / 2), orient - (width_angle / 2)]))
    try: angle[angle < 0.] = angle[angle < 0.] + 360
    except: pass
    try: angle[angle > 360.] = angle[angle > 360.] - 360
    except: pass
    degrees = 90 - np.rad2deg(np.arctan2(y - qb_position[:, 1], x - qb_position[:, 0]))
    try: degrees[degrees < 0.] = degrees[degrees < 0.] + 360
    except: pass
    try: degrees[degrees > 360.] = degrees[degrees < 0.] - 360
    except: pass
    
    cond1_ = 1 * (np.round(np.max(angle, axis = 1) - np.min(angle, axis = 1),0) == width_angle)
    cond2_ = 1 - cond1_
    
    return cond1_ * ((degrees >= np.min(angle, axis = 1)) * (degrees <= np.max(angle, axis = 1)) * _circ) + \
        cond2_ * (((degrees <= np.min(angle, axis = 1)) + (degrees >= np.max(angle, axis = 1))) * _circ)


def helmet_cover(mat_, width_, height_):
    """
    Function that returns a matrix of 0 and 1 that plotted look like the
    helmet of a QB from his POV
    Parameters
    ----------
    mat_ : matrix
        matrix of zeros.
    width_ : int
        width of matrix.
    height_ : height
        height of matrix.
    Returns
    -------
    ret_ : matrix
        matrix of 0 and 1 that plotted look like the helmet of a QB.
    """
    if height_ == 300:
        x = mat_[:,0]; y = mat_[:,1]
        # Top
        _top = (x - 0.5 * width_)** 2 + (y + 1240)**2 >= (1500 ** 2)
        # Bars - angles
        _bar_1 = ((x - 0.5 * width_)** 2 + (y + 810)**2 <= (900 ** 2)) * \
            ((x - 0.5 * width_)** 2 + (y + 820)**2 >= (900 ** 2))
            
        _bar_2 = ((x - 0.5 * width_)** 2 + (y + 840)**2 <= (900 ** 2)) * \
            ((x - 0.5 * width_)** 2 + (y + 850)**2 >= (900 ** 2))
            
        _bar_3 = (x - 0.5 * width_)** 2 + (y + 890)**2 <= (900 ** 2)
        # Bars - vertical
        _bar_4 = ((x - 0.5 * width_)** 2 + (y + 810)**2 <= (900 ** 2)) * ((y + 15 * x) >= 2000) * \
            ((y + 15 * x) <= 2140)
        _bar_5 = ((x - 0.5 * width_)** 2 + (y + 810)**2 <= (900 ** 2)) * ((y - 15 * x) >= -5785) * \
            ((y - 15 * x) <= -5645)
        _top = _top * 1; _bar_1 = _bar_1 * 1; _bar_2 = _bar_2 * 1;
        _bar_3 = _bar_3 * 1; _bar_4 = _bar_4 * 1; _bar_5 = _bar_5 * 1;
        ret_ = _top + _bar_1 + _bar_2 + _bar_3 + _bar_4 + _bar_5
        return ret_
    
    elif height_ == 100:
        x = mat_[:,0]; y = mat_[:,1]
        # Top
        _top = (x - 0.5 * width_)** 2 + (y + 210)**2 >= (300 ** 2)
        # Bars - angles
        _bar_1 = ((x - 0.5 * width_)** 2 + (y + 270)**2 >= (300 ** 2)) * \
            ((x - 0.5 * width_)** 2 + (y + 267)**2 <= (300 ** 2))
            
        _bar_2 = ((x - 0.5 * width_)** 2 + (y + 280)**2 >= (300 ** 2)) * \
            ((x - 0.5 * width_)** 2 + (y + 277)**2 <= (300 ** 2))
            
        _bar_3 = (x - 0.5 * width_)** 2 + (y + 295)**2 <= (300 ** 2)
        # Bars - vertical
        _bar_4 = ((x - 0.5 * width_)** 2 + (y + 267)**2 <= (300 ** 2)) * ((y + 15 * x) >= 700) * \
            ((y + 15 * x) <= 740)
        _bar_5 = ((x - 0.5 * width_)** 2 + (y + 267)**2 <= (300 ** 2)) * ((y - 15 * x) >= -1895) * \
            ((y - 15 * x) <= -1855)
        _top = _top * 1; _bar_1 = _bar_1 * 1; _bar_2 = _bar_2 * 1;
        _bar_3 = _bar_3 * 1; _bar_4 = _bar_4 * 1; _bar_5 = _bar_5 * 1;
        ret_ = _top + _bar_1 + _bar_2 + _bar_3 + _bar_4 + _bar_5
        return ret_
    
    

def equation_line_two_points(point0, point1):
    m = (point1[1] - point0[1]) / (point1[0] - point0[0])
    m2 = (m * point0[0] - point0[1]) / m
    return m, m2

def monigote(xm_ym, perc, height_, width_, mat_):
    
    # xrange = np.arange(0, width_)
    # yrange = np.arange(0, height_)
    # xx, yy = np.meshgrid(xrange, yrange)
    # mat_ = np.c_[xx.ravel(), yy.ravel()]
    
    x = mat_[:,0]
    y = mat_[:,1]
    
    h_ = int(np.round(height_ * perc[0], 0))
    w_ = int(np.round(width_ * perc[1], 0))
    
    xmed_ymed = np.array([int(np.round(xm_ym[0] * width_, 0)),
                          int(np.round(xm_ym[1] * height_, 0))])
        
    # Circulo ---
    centro_circ = np.array([xmed_ymed[0], xmed_ymed[1] + (h_ / 2) * 8.25 / 10])
    radio_circ = (h_ / 2) * 1.75 / 10
    head = (x - centro_circ[0]) ** 2 + (y - centro_circ[1]) **2 <= radio_circ ** 2
    # shoulder = y <= xmed_ymed[1] + (h_ / 2) * 7.25 / 10
    s_1 = np.array([xmed_ymed[0] - w_ / 2 + w_ / 10,
                           xmed_ymed[1] + (h_ / 2) * 7.25 / 10 - w_ / 10])
    s_2 = np.array([xmed_ymed[0] + w_ / 2 - w_ / 10,
                           xmed_ymed[1] + (h_ / 2) * 7.25 / 10 - w_ / 10])
    radio_shoulder = w_/10
    shoulder_1 = (x - s_1[0]) ** 2 + (y - s_1[1]) **2 <= radio_shoulder ** 2
    shoulder_2 = (x - s_2[0]) ** 2 + (y - s_2[1]) **2 <= radio_shoulder ** 2
    
    #A
    shoulder = y <= xmed_ymed[1] + (h_ / 2) * 7.25 / 10
    half_shoulder = y >= xmed_ymed[1] + (h_ / 2) * 7.25 / 10 - w_ / 10
    half_shoulder_x_1 = x >= xmed_ymed[0] - w_ / 2 + w_ / 10
    half_shoulder_x_2 = x <= xmed_ymed[0] + w_ / 2 - w_ / 10
    upper_body = (1*shoulder) * (1*half_shoulder) * (1*half_shoulder_x_1) * (1*half_shoulder_x_2)
    upper_body[upper_body!=0] = 1
    
    #B
    armpit = y >= xmed_ymed[1] + (h_ / 2) * 3 / 10
    half_shoulder_2 = y <= xmed_ymed[1] + (h_ / 2) * 7.25 / 10 - w_ / 10
    side_1 = x >= xmed_ymed[0] - w_ / 2
    side_2 = x <= xmed_ymed[0] + w_ / 2
    upper_body = upper_body + ((1*armpit) * (1*half_shoulder_2) * (1*side_1) * (1*side_2))
    upper_body[upper_body!=0] = 1
    
    #C
    upper_body = upper_body + (1*shoulder_1) + (1*shoulder_2)
    upper_body[upper_body!=0] = 1
    
    #D
    upper_body = upper_body + (1*head)
    upper_body[upper_body!=0] = 1
    
    #E
    feet_1 = np.array([xmed_ymed[0] - (1 / 3) * w_ / 2,
                       xmed_ymed[1] - h_ / 2])
    feet_2 = np.array([xmed_ymed[0] + (1 / 3) * w_ / 2,
                       xmed_ymed[1] - h_ / 2])
    arm_inter_1 = np.array([xmed_ymed[0] - w_ / 2 + w_ / 10,
                            xmed_ymed[1] + (h_ / 2) * 3 / 10])
    arm_inter_2 = np.array([xmed_ymed[0] + w_ / 2 - w_ / 10,
                            xmed_ymed[1] + (h_ / 2) * 3 / 10])
    
    m_1, m_11 = equation_line_two_points(arm_inter_1, feet_1)
    m_1_eq = x >= (y / m_1) + m_11
    m_2, m_21 = equation_line_two_points(arm_inter_2, feet_2)
    m_2_eq = x <= (y / m_2) + m_21
    
    armpit_2 = y <= xmed_ymed[1] + (h_ / 2) * 3 / 10
    feet = y >= xmed_ymed[1] - (h_ / 2)
    upper_body = upper_body + ((1*m_1_eq) * (1*m_2_eq) * (1*armpit_2) * (1*feet))
    upper_body[upper_body!=0] = 1
    
    
    
    upper_body = upper_body.reshape((height_, width_)) * 1
 

    
    # feet = y >= xmed_ymed[1] - (h_ / 2)
    
    # side_i = x >= xmed_ymed[0] - (w_ / 2)
    # side_d = x <= xmed_ymed[0] + (w_ / 2)
    # body = (1 * side_i) * (1 * side_d) * (1 * shoulder) * (1 * feet)
    # shape = (head * 1) + body
    # shape[shape != 0] = 1
    # shape = shape.reshape(xx.shape) * 1
    
    return upper_body

def visual_angle_perc_calculator(dataframe, w_, h_, m_helmet, mat_):
    #Helmet_cover_perc = 0.20108121141975308 #Redo in future
    
    dataframe_df = dataframe[dataframe['team'] != dataframe['team_QB'].iloc[0]].copy()
    dataframe_of = dataframe[dataframe['team'] == dataframe['team_QB'].iloc[0]].copy()
    
    # dataframe_df = dataframe_df.sort_values(by = 'distance').copy()
    # dataframe_of = dataframe_of.sort_values(by = 'distance').copy()
    
    dataframe = pd.concat([dataframe_df, dataframe_of])
    dataframe = dataframe[dataframe['team'] != 'football'].copy()
    # xrange = np.arange(0, w_)
    # yrange = np.arange(0, h_)
    # xx, yy = np.meshgrid(xrange, yrange)
    # mat_ = np.c_[xx.ravel(), yy.ravel()]
    
    # m_helmet = helmet_cover(mat_, w_, h_)
    # m_helmet = m_helmet.reshape(xx.shape)
    # m_helmet[m_helmet != 0] = 1
    
    dataframe = dataframe.sort_values(by = 'distance').copy()

    m_ = m_helmet.copy()
    total_view_w_helmet = 1 - (m_.sum() / (m_.shape[0] * m_.shape[1]))
    
    for i in range(0, len(dataframe)):
        if dataframe.loc[dataframe.index[i], 'in_angle'] != 0:
            xm_ym = np.array([dataframe.loc[dataframe.index[i], 'xmed'],
                              dataframe.loc[dataframe.index[i], 'y_s'] + 
                              0.5 * (dataframe.loc[dataframe.index[i], 'y_t'] -
                              dataframe.loc[dataframe.index[i], 'y_s'])])
            perc = np.array([dataframe.loc[dataframe.index[i], 'perc_height'],
                                  dataframe.loc[dataframe.index[i], 'perc_width']])
            single_ = monigote(xm_ym, perc, h_, w_, mat_)
            m_ = m_ + single_
            m_[m_!=0] = 1
            
            single_helmet_ = single_+ m_helmet
            single_helmet_[single_helmet_ != 0] = 1
            
            
            dataframe.loc[dataframe.index[i], 'perc_visual'] = (m_.sum() / (m_.shape[0] * m_.shape[1])) - (1 - total_view_w_helmet)
            total_view_w_helmet = 1 - (m_.sum() / (m_.shape[0] * m_.shape[1]))
            dataframe.loc[dataframe.index[i], 'perc_visual_single'] = single_helmet_.sum() / (single_helmet_.shape[0] * single_helmet_.shape[1]) - \
                m_helmet.sum() / (m_helmet.shape[0] * m_helmet.shape[1])
            # total_cover = m_.sum() / (m_.shape[0] * m_.shape[1])
    
    dataframe['perc_visual_total'] = total_view_w_helmet
    
    return dataframe



def plotlines(play):
    fig = go.Figure()
    # line_scrim = dict(color = '#008013', width = 1)
    # fig.add_trace(go.Scatter(x = np.repeat(play['yardlineNumber'].iloc[0], 53.4 * 10),
    #                          y = np.arange(0, 53.4, 0.1),
    #                          mode='lines',
    #                          line = line_scrim))
    # fig.add_trace(go.Scatter(x = np.repeat(play['yardlineNumber'].iloc[0] + 
    #                                        play['yardsToGo'].iloc[0], 53.4 * 10),
    #                          y = np.arange(0, 53.4, 0.1),
    #                          mode='lines',
    #                          line = line_scrim,
    #                          fill='tonexty', 
    #                          fillcolor = '#008013'))
    return fig

def plotfield(fig):
    
    line_side = dict(color = 'white', width = 2)
    line_middle = dict(color = 'white', width = 1)
    line_small = dict(color = 'white', width = 1)
    
    fig.add_trace(go.Scatter(x = np.arange(10, 111), y = np.repeat(0, 101), mode='lines', line = line_side))
    fig.add_trace(go.Scatter(x = np.arange(10, 111), y = np.repeat(53.3, 111), mode='lines', line = line_side))
    fig.add_trace(go.Scatter(x = np.repeat(10, 53.4 * 10), y = np.arange(0, 53.4, 0.1), mode='lines', line = line_side))
    fig.add_trace(go.Scatter(x = np.repeat(110, 53.4 * 10), y = np.arange(0, 53.4, 0.1), mode='lines', line = line_side))
    
    for i in range(12, 110, 2):
        fig.add_trace(go.Scatter(x = np.repeat(i, 53.4 * 10),
                                 y = np.arange(0, 2, 0.1),
                                 mode = 'lines',
                                 line = line_small))
        fig.add_trace(go.Scatter(x = np.repeat(i, 53.4 * 10),
                                  y = np.arange(51.4, 53.4, 0.1),
                                  mode = 'lines',
                                  line = line_small))
    for i in range(20, 110, 10):
        fig.add_trace(go.Scatter(x = np.repeat(i, 53.4 * 10),
                                 y = np.arange(0, 53.4, 0.1),
                                 mode = 'lines',
                                 line = line_middle))
    
    fig.update_layout(margin = dict(l = 0,
                                    r = 0,
                                    t = 0,
                                    b = 0))
    fig.update_xaxes(title = None,
                     showticklabels = False,
                     showgrid = False,
                     zeroline = False)
    fig.update_xaxes({'range': (0, 120), 'autorange': False})
    fig.update_yaxes(title = None,
                     showticklabels = False,
                     showgrid = False,
                     zeroline = False)
    fig.update_layout(paper_bgcolor='#009A17', plot_bgcolor='#009A17')
    fig.update_layout(showlegend = False)
    return fig

def check_angle_reach(qb_position, radio, def_position, orient, width_angle):
    """
    Check if defender is within angle reach of the QB
    Parameters
    ----------
    qb_position : np.array
        X, y coordinates of QB.
    radio : integer
        radio of vision of QB.
    array : coordinate/s of defenders
        array.
    orient : float
        angle orientation of QB when passing.

    Returns
    -------
    TYPE: Bool, wheter it is inside or not of the QBs angle vision
    """
    try: # If def_position is vector
        x = def_position[:,0]
        y = def_position[:,1]
    except: # If def_position is single value
        x = def_position[0]
        y = def_position[1]
        
    # First the circle
    _circ =  (x - qb_position[0])** 2 + (y - qb_position[1]) ** 2 <= radio ** 2
    # Second the arc

    angle = np.array([orient + (width_angle / 2), orient - (width_angle / 2)])
    try: angle[angle < 0.] = angle[angle < 0.] + 360
    except: pass
    try: angle[angle > 360.] = angle[angle > 360.] - 360
    except: pass
    degrees = 90 - np.rad2deg(np.arctan2(y - qb_position[1], x - qb_position[0]))
    
    try: # If def_position is vector
        if degrees < 0:
            degrees = degrees + 360
        elif degrees > 360:
            degrees = degrees - 360
    except: # If def_position is single value
        degrees[degrees < 0.] = degrees[degrees < 0.] + 360
        degrees[degrees > 360.] = degrees[degrees < 0.] - 360

    if int(np.round(max(angle) - min(angle), 0)) == width_angle:
        deg_min = degrees >= min(angle)
        deg_max = degrees <= max(angle)
        def_final = deg_min * deg_max * _circ

    else:
        deg_min = degrees <= min(angle)
        deg_max = degrees >= max(angle)
        def_final = (deg_min + deg_max) * _circ
    
    return def_final



def plotangle(fig, play_action, off, deff, data_players):
    offense = play_action[play_action['team'] == off].copy()
    offense = offense.merge(data_players[['nflId', 'officialPosition']],
                            left_on = 'nflId', right_on = 'nflId', 
                            how = 'left').copy()
    offense = offense[offense['officialPosition'] == 'QB'].copy()
    
    cent = np.array([offense['x'].iloc[0], offense['y'].iloc[0]])
    xrange = np.round(np.arange(cent[0] - 110, cent[0] + 110, 0.05), 2)
    yrange = np.round(np.arange(cent[1] - 110, cent[1] + 110, 0.05), 2)

    xrange = xrange[(xrange >= 0) & (xrange <= 120)]
        
    yrange = yrange[(yrange >= 0) & (yrange <= 53.3)]
    xx, yy = np.meshgrid(xrange, yrange)

    _dist = 30
    
    # Z = circle(cent, _dist + 1, np.c_[xx.ravel(), yy.ravel()], offense['o'].iloc[0])
    Z = check_angle_reach(cent, _dist + 1, np.c_[xx.ravel(), yy.ravel()], 
                      offense['o'].iloc[0], 120)
    Z = Z.reshape(xx.shape)
    Z = np.array(Z, dtype = int)
    
    fig.add_trace(go.Contour(
        x = xrange, y = yrange, z = Z,
        connectgaps = True,
        colorscale = ['#009A17', 'red'],
        showscale = False, opacity = 0.25,
    ))
    return fig
    
def plotplayers(fig, play_action, off, deff, data_players):
    offense = play_action[play_action['team'] == off].copy()
    offense = offense.merge(data_players[['nflId', 'officialPosition']],
                            left_on = 'nflId', right_on = 'nflId', 
                            how = 'left').copy()
    offense['is_QB'] = 0
    offense.loc[offense['officialPosition'] == 'QB', 'is_QB']  = 1
    fig.add_trace((go.Scatter(x = offense['x'],
                              y = offense['y'], 
                              mode = 'markers',
                              marker=dict(color = offense['is_QB'],
                                          symbol = offense['is_QB'],
                                          size = 8,
                                          line = dict(color='black', width = 2),
                                          colorscale = ['red', 'red']))))
    deffense = play_action[play_action['team'] == deff].copy()
    fig.add_trace((go.Scatter(x = deffense['x'],
                              y = deffense['y'], 
                              mode = 'markers',
                              marker=dict(size = 8, line = dict(color='black', width = 2),
                                          color = '#009dff'))))
    return fig

def plotplayersdl(fig, play_action, off, deff, data_players, _data):
    offense = play_action[play_action['team'] == off].copy()
    offense = offense.merge(data_players[['nflId', 'officialPosition']],
                            left_on = 'nflId', right_on = 'nflId', 
                            how = 'left').copy()
    offense['is_QB'] = 0
    offense.loc[offense['officialPosition'] == 'QB', 'is_QB']  = 1
    offense_qb = offense[offense['officialPosition'] == 'QB'].copy()
    
    
    cent = np.array([offense_qb['x'].iloc[0], offense_qb['y'].iloc[0]])
    xrange = np.round(np.arange(cent[0] - 110, cent[0] + 110, 0.1), 2)
    yrange = np.round(np.arange(cent[1] - 110, cent[1] + 110, 0.1), 2)

    xrange = xrange[(xrange >= 0) & (xrange <= 120)]
        
    yrange = yrange[(yrange >= 0) & (yrange <= 53.3)]
    xx, yy = np.meshgrid(xrange, yrange)
    
    _data_a = _data.copy()
    _data_a = _data_a[_data_a['team'] == deff].copy()

    X = np.zeros(shape = (len(_data_a),2))
    X[:,0] = _data_a['x']
    X[:,1] = _data_a['y']
    y = np.arange(0, len(_data_a))
    
    
    knn = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform')
    knn.fit(X, y)

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    # 
    _dist = 30
    Z2 = check_angle_reach(cent, _dist + 1, np.c_[xx.ravel(), yy.ravel()], 
                      offense_qb['o'].iloc[0], 120)
    Z2 = Z2.reshape(xx.shape)
    Z2 = np.array(Z2, dtype = int)
    
    
    Z = Z * Z2
    
    
    fig.add_trace(go.Contour(
        x = xrange, y = yrange, z = Z,
        connectgaps = True,
        colorscale = ['#009A17', 'red','yellow','blue'],
        showscale = False, opacity = 0.5,
    ))
    
    
    
    for i in range(len(_data)):
        if pd.isna(_data['closest_defender'].iloc[i]) == True:
            _data.loc[_data.index[i], 'closest_defender'] = _data['nflId'].iloc[i]
            
    for i in range(len(_data)):
        fig.add_trace((go.Scatter(x = _data['x'],
                              y = _data['y'], 
                              mode = 'markers',
                              marker=dict(color = _data['closest_defender'],
                                          size = 10,
                                          line = dict(color='black', width = 2)
                                          ))))
        fig.add_trace((go.Scatter(x = _data['x_QB'],
                                  y = _data['y_QB'],
                                  mode = 'markers',
                                  marker = dict(color = 'red',
                                                size = 10,
                                                symbol = 'square',
                                                line = dict(color='black', width = 2)))))
    return fig



def remove_players_delfos(dataframe):
    dataframe_df = dataframe[dataframe['team'] != dataframe['team_QB'].iloc[0]].copy()
    dataframe_df = dataframe_df[dataframe_df['officialPosition'].isin(['DE', 'DT', 'NT'])].copy()
    dataframe_df['distance_defender'] = np.nan
    dataframe_df['closest_defender'] = np.nan
    
    #Compute the closest defensive player for each offensive player. If distance > 200 cm to the closest
    #then delete
    
    dataframe_of = dataframe[dataframe['team'] == dataframe['team_QB'].iloc[0]].copy()
    dataframe_of['distance_defender'] = np.nan
    dataframe_of['closest_defender'] = np.nan
    
    for i in range(len(dataframe_of)):
        d = 200
        for j in range(len(dataframe_df)):
            if distance_vector(dataframe_of['x'].iloc[i], dataframe_df['x'].iloc[j],
                               dataframe_of['y'].iloc[i], dataframe_df['y'].iloc[j]) < d:
                d = distance_vector(dataframe_of['x'].iloc[i], dataframe_df['x'].iloc[j],
                                   dataframe_of['y'].iloc[i], dataframe_df['y'].iloc[j])
                dataframe_of.loc[dataframe_of.index[i],'closest_defender'] = dataframe_df['nflId'].iloc[j]
                dataframe_of.loc[dataframe_of.index[i],'distance_defender'] = d
    dataframe_df = dataframe_df.sort_values(by = 'distance').copy()
    dataframe_of = dataframe_of.sort_values(by = 'distance').copy()
    dataframe_of.dropna(inplace = True)
    
    dataframe = pd.concat([dataframe_df, dataframe_of])
    dataframe = dataframe[dataframe['team'] != 'football'].copy()
    dataframe = dataframe.sort_values(by = 'distance').copy()
    
    return [list(dataframe.index), list(dataframe['closest_defender'])]


def visual_angle_perc_calculator_linemen(dataframe, w_, h_, m_helmet, mat_):
    #Helmet_cover_perc = 0.20108121141975308 #Redo in future
    

    # xrange = np.arange(0, w_)
    # yrange = np.arange(0, h_)
    # xx, yy = np.meshgrid(xrange, yrange)
    # mat_ = np.c_[xx.ravel(), yy.ravel()]
    
    # m_helmet = helmet_cover(mat_, w_, h_)
    # m_helmet = m_helmet.reshape(xx.shape)
    # m_helmet[m_helmet != 0] = 1

    m_ = m_helmet.copy()
    total_view_w_helmet = 1 - (m_.sum() / (m_.shape[0] * m_.shape[1]))
    
    for i in range(0, len(dataframe)):
        if dataframe.loc[dataframe.index[i], 'in_angle'] != 0:
            xm_ym = np.array([dataframe.loc[dataframe.index[i], 'xmed'],
                              dataframe.loc[dataframe.index[i], 'y_s'] + 
                              0.5 * (dataframe.loc[dataframe.index[i], 'y_t'] -
                              dataframe.loc[dataframe.index[i], 'y_s'])])
            perc = np.array([dataframe.loc[dataframe.index[i], 'perc_height'],
                                  dataframe.loc[dataframe.index[i], 'perc_width']])
            single_ = monigote(xm_ym, perc, h_, w_, mat_)
            m_ = m_ + single_
            m_[m_!=0] = 1
            
            single_helmet_ = single_+ m_helmet
            single_helmet_[single_helmet_ != 0] = 1
            
            
            dataframe.loc[dataframe.index[i], 'perc_visual'] = (m_.sum() / (m_.shape[0] * m_.shape[1])) - (1 - total_view_w_helmet)
            total_view_w_helmet = 1 - (m_.sum() / (m_.shape[0] * m_.shape[1]))
            dataframe.loc[dataframe.index[i], 'perc_visual_single'] = single_helmet_.sum() / (single_helmet_.shape[0] * single_helmet_.shape[1]) - \
                m_helmet.sum() / (m_helmet.shape[0] * m_helmet.shape[1])
            # total_cover = m_.sum() / (m_.shape[0] * m_.shape[1])
    
    dataframe['perc_visual_total'] = total_view_w_helmet
    
    return dataframe



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4. Inputs
###############################################################################
h_angle = 45
w_angle = 60

height_ = 300
width_ = int(height_ * np.tan(np.deg2rad(60)))

read_loaded = True # Boolean if True, the main calculations will be loaded from 
                   # the saved files. If False all the calculations will be 
                   # performed.
                   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.1 Code - Dictionary to store plays by game and event
###############################################################################
game_play_id = {} # Dictionary with the plays for each game
for i in list(week.keys()):
    _gp_id = list(np.unique(week[i]['gameId']))
    for g in _gp_id:
        game_play_id[g] = list(np.unique(week[i][week[i]['gameId'] == g]['playId']))
# Create a master table with every pass play o every game containing 
# relevant information for the calculations.
game_play_df = pd.DataFrame()

if read_loaded == True:
    with open('data/game_play_df.pickle', 'rb') as handle:
        game_play_df = pickle.load(handle)
        
if read_loaded == False:
    for g in list(game_play_id.keys()):
        print(g)
        for p in game_play_id[g]:
            _week_t = week_t[week_t['gameId'] == g].copy()
            _week_t = _week_t[_week_t['playId'] == p].copy()
            if 'pass_forward' in np.unique(_week_t['event']): _ev = 'pass_forward'
            elif 'autoevent_passforward' in np.unique(_week_t['event']): _ev = 'autoevent_passforward'
            elif 'qb_sack' in np.unique(_week_t['event']): _ev = 'qb_sack'
            elif 'run' in np.unique(_week_t['event']): _ev = 'run'
            elif 'qb_strip_sack' in np.unique(_week_t['event']): _ev = 'qb_strip_sack'

            if 'ball_snap' in np.unique(_week_t['event']):
                # Ball in QB hands (in sec)
                _time_ball = (parser.parse(_week_t[_week_t['event'] == _ev]['time'].iloc[0]) - \
                              parser.parse(_week_t[_week_t['event'] == 'ball_snap']\
                                           ['time'].iloc[0])).total_seconds()
                _week_t = _week_t[_week_t['event'] == _ev].copy() 
                # Lets get the QB and the deffensive players (all)
                # 1) Lets find out who attacks
                _plays = plays[plays['gameId'] == g].copy()
                _plays = _plays[_plays['playId'] == p].copy()
                _offense = _plays['possessionTeam'].iloc[0]
                _defense = _plays['defensiveTeam'].iloc[0]
                # 2) Lets get the QB and deffense
                _week_t = _week_t.merge(players[['nflId', 'officialPosition']],
                                        left_on = 'nflId', right_on = 'nflId', 
                                        how = 'left').copy()
                # _week_t = _week_t[(_week_t['officialPosition'] == 'QB') |
                #            (_week_t['team'] == _defense)].copy()
                _week_t['is_qb'] = _week_t['officialPosition'] == 'QB'
                _week_t = _week_t.merge(_plays, how='left', 
                                        left_on=['gameId','playId'],
                                        right_on = ['gameId','playId']).copy()
                _week_t['qb_timeball'] = _time_ball
                game_play_df = pd.concat([game_play_df, _week_t]).copy()
                
    with open('data/game_play_df.pickle', 'wb') as handle:
        pickle.dump(game_play_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.2 Code - Recover saved information and calculate width, height and
#            location
###############################################################################
# Now that the master table is created, lets calculate the distance to the 
# the QB, and the space that takes in the POV of the QB
master_dataframe = game_play_df[['gameId', 'playId', 'nflId', 'officialPosition',
                             'playDirection', 'x', 'y', 'o', 'event', 'team']].copy()
master_dataframe = master_dataframe.merge(players[['nflId', 'height', 'weight']], 
                                  left_on = 'nflId', right_on = 'nflId',
                                  how = 'left').copy()
master_dataframe = intocm(master_dataframe, 'height').copy()
master_dataframe = master_dataframe.loc[master_dataframe['event'].isin(['autoevent_passforward',
                                                   'pass_forward']), :].copy()
master_dataframe['gameId'] = master_dataframe['gameId'].apply(lambda x: str(x))
master_dataframe['playId'] = master_dataframe['playId'].apply(lambda x: str(x))
master_dataframe['code'] = master_dataframe['gameId'] + '_' + master_dataframe['playId']
d_ = master_dataframe.groupby('code').agg({'x':'count'})
master_dataframe = master_dataframe.loc[master_dataframe['code'].isin(d_[d_['x'] == 23].index), :]
master_dataframe_qb = master_dataframe[master_dataframe['officialPosition'] == 'QB'].copy()
master_dataframe = master_dataframe[master_dataframe['officialPosition'] != 'QB'].copy()
master_dataframe = master_dataframe.merge(master_dataframe_qb[['code', 'nflId', 'x', 'y', 'o',
                                          'height', 'weight','officialPosition', 'team']],
                                          left_on = 'code', right_on = 'code', how = 'outer',
                                          suffixes=('', '_QB'))
master_dataframe['distance'] = distance_vector(master_dataframe['x_QB'], master_dataframe['x'],
                                        master_dataframe['y_QB'], master_dataframe['y']).values
master_dataframe['in_angle'] = check_angle_reach_2(np.transpose(np.array([master_dataframe['x_QB'],
                                                                          master_dataframe['y_QB']])), 1e6,
                                                   np.transpose(np.array([master_dataframe['x'],
                                                                          master_dataframe['y']])),
                                                   master_dataframe['o_QB'], int(w_angle * 2))
master_dataframe['eye_QB'] = master_dataframe['height_QB'] - 7
# Size of vision field
master_dataframe['ht_QB'] = 2 * np.tan(np.deg2rad(h_angle)) * master_dataframe['distance']
master_dataframe['wt_QB'] = 2 * np.tan(np.deg2rad(w_angle)) * master_dataframe['distance']
# Calculate width of defender
master_dataframe['wratio_QB'] = height_weight_to_width_new(master_dataframe['height'], master_dataframe['weight'])
master_dataframe['width'] = master_dataframe['height'] * master_dataframe['wratio_QB']
# Perc points
master_dataframe['perc_height'] = master_dataframe['height'] / master_dataframe['ht_QB']
master_dataframe['perc_width'] = master_dataframe['width'] / master_dataframe['wt_QB']
# Starting points
# 1) Height
master_dataframe['distancia_cross_down_QB'] = master_dataframe['eye_QB'] / np.tan(np.deg2rad(h_angle))
# A) distancia_defensa > distancia_cross_down
gamma = 90 - h_angle
_a = master_dataframe['eye_QB'] * np.tan(np.deg2rad(gamma))
_b = master_dataframe['distance'] - _a
h_s = np.tan(np.deg2rad(h_angle)) * _b
_cond = master_dataframe['distance'] > master_dataframe['distancia_cross_down_QB']
_cond_index = _cond[_cond == True].index
master_dataframe.loc[_cond_index, 'y_s'] = (h_s[_cond_index] / master_dataframe.loc[_cond_index, 'ht_QB'])
master_dataframe.loc[_cond_index, 'y_t'] = master_dataframe.loc[_cond_index, 'y_s'] + \
    master_dataframe.loc[_cond_index, 'perc_height']
#B) distancia_defensa == distancia_cross_down
_cond = master_dataframe['distance'] == master_dataframe['distancia_cross_down_QB']
_cond_index = _cond[_cond == True].index
master_dataframe.loc[_cond_index, 'y_s'] = 0 
master_dataframe.loc[_cond_index, 'y_t'] = (master_dataframe.loc[_cond_index, 'y_s'] + \
                                            master_dataframe.loc[_cond_index, 'perc_height'])
#C) distancia_defensa < distancia_cross_down
_cond = master_dataframe['distance'] < master_dataframe['distancia_cross_down_QB']
_cond_index = _cond[_cond == True].index
d_plus = master_dataframe.loc[_cond_index, 'distancia_cross_down_QB'] - \
    master_dataframe.loc[_cond_index, 'distance']
h_plus = d_plus * np.tan(np.deg2rad(h_angle))
master_dataframe.loc[_cond_index, 'y_s'] = 0
master_dataframe.loc[_cond_index, 'y_t'] = (master_dataframe.loc[_cond_index, 'height'] - h_plus) / \
    master_dataframe.loc[_cond_index, 'ht_QB']
f_0 = np.transpose(np.array([master_dataframe['x_QB'], master_dataframe['y_QB']]))
f_def = np.transpose(np.array([master_dataframe['x'], master_dataframe['y']]))
f_ = two_points_point_angle_2(f_0, master_dataframe['o_QB'].values) # Focal point
angle = np.transpose(np.array([master_dataframe['o_QB'].copy() + w_angle, 
                               master_dataframe['o_QB'].copy() - w_angle]))

angle[angle > 360] = angle[angle > 360] - 360
angle[angle < 0] = angle[angle < 0] + 360
f_l = np.transpose(np.array([master_dataframe['x_QB'], master_dataframe['y_QB']])) * 0 # Structure
f_u = np.transpose(np.array([master_dataframe['x_QB'], master_dataframe['y_QB']])) * 0 # Structure

_ind = pd.Series((np.max(angle, axis = 1) - np.min(angle, axis = 1)).astype(int) == int(w_angle * 2))
f_l[_ind[_ind == True].index] = two_points_point_angle_2(f_0[_ind[_ind == True].index],
                                                         np.min(angle, axis = 1)[_ind[_ind == True].index])
f_u[_ind[_ind == True].index] = two_points_point_angle_2(f_0[_ind[_ind == True].index],
                                                         np.max(angle, axis = 1)[_ind[_ind == True].index])

f_l[_ind[_ind != True].index] = two_points_point_angle_2(f_0[_ind[_ind != True].index],
                                                         np.max(angle, axis = 1)[_ind[_ind != True].index])
f_u[_ind[_ind != True].index] = two_points_point_angle_2(f_0[_ind[_ind != True].index],
                                                         np.min(angle, axis = 1)[_ind[_ind != True].index])

f_def_l = perp_points_2(f_0, f_, f_def, master_dataframe['o_QB'].values)
# Intersections
inter_1 = intersection_vectorized(f_0, f_u, f_def, f_def_l)
inter_2 = intersection_vectorized(f_0, f_l, f_def, f_def_l)

_ind = pd.Series((np.max(angle, axis = 1) - np.min(angle, axis = 1)).astype(int) == int(w_angle * 2))
# i = _ind[_ind != True].index
# a = distance_vector(inter_1[i, 0], f_def[i, 0], inter_1[i, 1], f_def[i, 1]) / \
#     distance_vector(inter_1[i, 0], inter_2[i, 0], inter_1[i, 1], inter_2[i, 1])
# master_dataframe.loc[_ind[_ind != True].index, 'xmed'] = a

i = _ind[_ind != True].index
a = distance_vector(inter_2[i, 0], f_def[i, 0], inter_2[i, 1], f_def[i, 1]) / \
    distance_vector(inter_1[i, 0], inter_2[i, 0], inter_1[i, 1], inter_2[i, 1])

master_dataframe.loc[_ind[_ind != True].index, 'xmed'] = a
i = _ind[_ind == True].index
a = distance_vector(inter_2[i, 0], f_def[i, 0], inter_2[i, 1], f_def[i, 1]) / \
    distance_vector(inter_1[i, 0], inter_2[i, 0], inter_1[i, 1], inter_2[i, 1])

master_dataframe.loc[_ind[_ind == True].index, 'xmed'] = a
master_dataframe['x0'] = (master_dataframe['xmed'] - 0.5 * master_dataframe['perc_width']) 
master_dataframe['x1'] = (master_dataframe['xmed'] + 0.5 * master_dataframe['perc_width'])
master_dataframe.loc[master_dataframe[master_dataframe['x0'] < 0].index, 'x0'] = 0
master_dataframe.loc[master_dataframe[master_dataframe['x0'] > 1].index, 'x0'] = 1
master_dataframe['x0'] = master_dataframe['x0'] * master_dataframe['in_angle']
master_dataframe['x1'] = master_dataframe['x1'] * master_dataframe['in_angle']
master_dataframe['xmed'] = master_dataframe['xmed'] * master_dataframe['in_angle']
master_dataframe['y_s'] = master_dataframe['y_s'] * master_dataframe['in_angle']
master_dataframe['y_t'] = master_dataframe['y_t'] * master_dataframe['in_angle']

master_dataframe_delfos = master_dataframe.copy()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.3 Code - Calculate FOVO for all players
#
###############################################################################
height_ = 300
width_ = int(height_ * np.tan(np.deg2rad(60)))

xrange = np.arange(0, width_)
yrange = np.arange(0, height_)
xx, yy = np.meshgrid(xrange, yrange)
mat_ = np.c_[xx.ravel(), yy.ravel()]
m_helmet = helmet_cover(mat_, width_, height_)
m_helmet = m_helmet.reshape(xx.shape)
m_helmet[m_helmet != 0] = 1



master_dataframe['perc_visual'] = 0
master_dataframe['perc_visual_single'] = 0
master_dataframe['perc_visual_total'] = 0

if read_loaded == True:
    with open('data/master_dataframe.pickle', 'rb') as handle:
        master_dataframe = pickle.load(handle)
if read_loaded == False:
    for g in list(np.unique(master_dataframe['gameId'])):
        _d_t = master_dataframe[master_dataframe['gameId'] == str(g)].copy()
        print(g)
        for p in list(np.unique(_d_t['playId'])):
            _data = _d_t[_d_t['playId'] == str(p)].copy()
            _data = visual_angle_perc_calculator(_data, width_, height_, m_helmet, mat_).copy()
            master_dataframe.loc[_data.index, 'perc_visual'] = _data['perc_visual'].values * \
                _data['in_angle'].values
            master_dataframe.loc[_data.index, 'perc_visual_single'] = _data['perc_visual_single'].values * \
                _data['in_angle'].values
            master_dataframe.loc[_data.index, 'perc_visual_total'] = _data['perc_visual_total'].values

    with open('data/master_dataframe.pickle', 'wb') as handle:
        pickle.dump(master_dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.4 Code - Plot 
#
###############################################################################
game_id = g = 2021091300
play_id = p = 902

g_list = [2021091300]
p_list = [902, 1075, 1174, 1226, 1250, 1293, 1344, 1368]
##########################

# Filter data tables for plotting # 2D
_week_t = week_t[week_t['gameId'] == game_id].copy()
_week_t = _week_t[_week_t['playId'] == play_id].copy()
_week_t = _week_t[_week_t['event'] == 'pass_forward'].copy()

_plays = plays[plays['gameId'] == game_id].copy()
_plays = _plays[_plays['playId'] == play_id].copy()

_scout = scout[scout['gameId'] == game_id].copy()
_scout = _scout[_scout['playId'] == play_id].copy()

_offense = _plays['possessionTeam'].iloc[0]
_defense = _plays['defensiveTeam'].iloc[0]
# Plot
figure = plotlines(_plays)
figure = plotfield(figure)
figure = plotangle(figure, _week_t, _offense, _defense, players)
figure = plotplayers(figure, _week_t, _offense, _defense, players)
figure.show()

#3D graph
position = list(itertools.product(list(np.arange(1, 3)), list(np.arange(1, 5))))
cont = 0
fig = make_subplots(2, 4, vertical_spacing = 0.01, horizontal_spacing= 0.01)
for i in range(len(g_list)):
    _g = g_list[i]
    for j in range(len(p_list)):
        _p = p_list[j]
        _d_t = master_dataframe[master_dataframe['gameId'] == str(_g)].copy()
        _data = _d_t[_d_t['playId'] == str(_p)].copy()
        _data = _data.sort_values(by = 'distance').copy()
        _data = _data[_data['team'] != 'football'].copy()
        m_ = m_helmet * 3
        for i in range(0, len(_data)):
            if _data.loc[_data.index[i], 'in_angle'] != 0:
                xm_ym = np.array([_data.loc[_data.index[i], 'xmed'],
                                  _data.loc[_data.index[i], 'y_s'] + 
                                  0.5 * (_data.loc[_data.index[i], 'y_t'] -
                                  _data.loc[_data.index[i], 'y_s'])])
                perc = np.array([_data.loc[_data.index[i], 'perc_height'],
                                      _data.loc[_data.index[i], 'perc_width']])
                single_ = monigote(xm_ym, perc, height_, width_, mat_)
                if _data['team'].iloc[i] == _data['team_QB'].iloc[i]:
                    m_[(m_ == 0) & (single_ == 1)] = 1
                else:
                    m_[(m_ == 0) & (single_ == 1)] = 2
                # m_[m_!=0] = 1
        # m_ = m_ + m_helmet * 10
        fig.add_trace(
            go.Heatmap(z = m_, colorscale = ['#e6e6e6','red', '#009dff', 'black']), position[cont][0], position[cont][1])
        cont = cont + 1

for row_ in range(1,3):
    for col_ in range(1,5):
        fig.update_xaxes(showgrid = False, row = row_, col = col_)
        fig.update_yaxes(showgrid = False, row = row_, col = col_)
        fig.update_traces(showscale = False, row=row_, col = col_)
        fig.update_xaxes(zeroline = True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_yaxes(zeroline = True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_yaxes(showline = True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels = False, row=row_, col=col_)
        fig.update_yaxes(showticklabels = False, row=row_, col=col_)


fig.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.5 Code - Analisis of image percentage accuracy passing 
#
###############################################################################
res_ana = master_dataframe.groupby('code').agg({'gameId': 'last',
                                                'playId': 'last',
                                                'nflId': 'last',
                                                'nflId_QB': 'last',
                                                'perc_visual_total': 'max'})

res_plays = plays[['gameId', 'playId', 'prePenaltyPlayResult', 'passResult']].copy()
res_plays['gameId'] = res_plays['gameId'].apply(lambda x: str(x))
res_plays['playId'] = res_plays['playId'].apply(lambda x: str(x))
res_plays['code'] = res_plays['gameId'] + '_' + res_plays['playId']

res_ana = res_ana.merge(res_plays, how='left', 
                        left_index = True, right_on = 'code').copy()

res_ana = res_ana[(res_ana['passResult'] == 'C') | (res_ana['passResult'] == 'I')].copy()
res_ana.loc[res_ana[res_ana['passResult'] == 'C'].index, 'passResult'] = 1
res_ana.loc[res_ana[res_ana['passResult'] == 'I'].index, 'passResult'] = 0

a = np.arange(0,1, 0.001)
df_2 = pd.DataFrame(a)
df_2.columns = ['perc_visual_total']
df_2['perc_acierto'] = 0
df_2['amount'] = 0

for i in range(0, len(df_2)):
    temp = res_ana[(res_ana['perc_visual_total'] <= df_2['perc_visual_total'].iloc[i])].copy()
    try:
        df_2.loc[df_2.index[i], 'perc_acierto'] = temp['passResult'].sum() / len(temp['passResult'])
    except:
        df_2.loc[df_2.index[i], 'perc_acierto'] = 0
    df_2.loc[df_2.index[i], 'amount'] = len(temp['passResult'])
df_2['amount'] = df_2['amount'] 

df_2['perc_visual_occ'] = 1 - df_2['perc_visual_total']

fig = px.line(x = 100 * df_2['perc_visual_total'], y = 100 * df_2['perc_acierto'],render_mode="svg")

# Set y-axes titles
fig.update_yaxes(title_text="Pass accuracy (%)")
fig.update_xaxes(title_text="Visual Field Available")
# fig.update_yaxes(title_text="Pass accuracy (%)", secondary_y = True)
fig.update_xaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_yaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_layout(paper_bgcolor = 'white',
               plot_bgcolor = 'white')
fig.update_layout(xaxis_ticksuffix = '%', xaxis_tickprefix = '<', yaxis_tickformat = ',.')
fig.update_layout(yaxis_ticksuffix = '%',yaxis_tickformat = ',.')
# fig.update_layout(yaxis2_ticksuffix = '%', yaxis_tickformat = ',.')
# 
fig.update_traces(line_color = '#5a6e8b', line=dict(width=6))
fig.update_layout( # customize font and legend orientation & position
    font_family = "Georgia",
    font_color="black",
    font_size = 24,
    legend=dict(
        font_size = 24, title=None, orientation = 'h', yanchor="bottom",
        y=1, x = 0.01, xanchor="left"))

# fig.update_layout(legend = dict(bgcolor = 'rgba(255,255,255,0.1)'))
fig.show()


fig = px.line(x = 100 * df_2['perc_visual_total'], y =df_2['amount'] )
fig.update_yaxes(title_text="Number of passes")
fig.update_xaxes(title_text="Visual Field Available")
# fig.update_yaxes(title_text="Pass accuracy (%)", secondary_y = True)
fig.update_xaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_yaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_layout(paper_bgcolor = 'white',
               plot_bgcolor = 'white')
fig.update_layout(xaxis_ticksuffix = '%', xaxis_tickprefix = '<', yaxis_tickformat = ',.')
# fig.update_layout(yaxis_ticksuffix = '%',yaxis_tickformat = ',.')
# fig.update_layout(yaxis2_ticksuffix = '%', yaxis_tickformat = ',.')
# 
fig.update_traces(line_color = '#5a6e8b', line=dict(width=6))
fig.update_layout( # customize font and legend orientation & position
    font_family = "Georgia",
    font_color="black",
    font_size = 24,
    legend=dict(
        font_size = 24, title=None, orientation = 'h', yanchor="bottom",
        y=1, x = 0.01, xanchor="left"))

# fig.update_layout(legend = dict(bgcolor = 'rgba(255,255,255,0.1)'))
fig.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5.3 Code - Calculate DELFOS
#
###############################################################################
height_ = 300
width_ = int(height_ * np.tan(np.deg2rad(60)))

xrange = np.arange(0, width_)
yrange = np.arange(0, height_)
xx, yy = np.meshgrid(xrange, yrange)
mat_ = np.c_[xx.ravel(), yy.ravel()]
m_helmet = helmet_cover(mat_, width_, height_)
m_helmet = m_helmet.reshape(xx.shape)
m_helmet[m_helmet != 0] = 1

master_dataframe_delfos['perc_visual'] = 0
master_dataframe_delfos['perc_visual_single'] = 0
master_dataframe_delfos['perc_visual_total'] = 0

#First remove the players from all the plays that are not involved in the 
#DELFOS construction
index_total = []
closest_def = []

if read_loaded == True:
    with open('data/master_dataframe_delfos.pickle', 'rb') as handle:
        master_dataframe_delfos = pickle.load(handle)
if read_loaded == False:
    for g in list(np.unique(master_dataframe_delfos['gameId'])):
        _d_t = master_dataframe[master_dataframe_delfos['gameId'] == str(g)].copy()
        for p in list(np.unique(_d_t['playId'])):
            _data = _d_t[_d_t['playId'] == str(p)].copy()
            index_partial = remove_players_delfos(_data)[0]
            closest_partial = remove_players_delfos(_data)[1]
            index_total.append(index_partial)
            closest_def.append(closest_partial)

    flat_list = []
    for sublist in index_total:
        for item in sublist: flat_list.append(item)
    
    flat_list_def = []
    for sublist in closest_def:
        for item in sublist: flat_list_def.append(item)

    master_dataframe_delfos = master_dataframe_delfos.loc[flat_list, :].copy()
    master_dataframe_delfos['closest_defender'] = flat_list_def
    
    # Now calculate delfos
    for g in list(np.unique(master_dataframe_delfos['gameId'])):
        _d_t = master_dataframe_delfos[master_dataframe_delfos['gameId'] == str(g)].copy()
        for p in list(np.unique(_d_t['playId'])):
            _data = _d_t[_d_t['playId'] == str(p)].copy()
            _data = visual_angle_perc_calculator_linemen(_data, width_, height_, m_helmet, mat_)
            master_dataframe_delfos.loc[_data.index, 'perc_visual'] = _data['perc_visual'].values * \
                _data['in_angle'].values
            master_dataframe_delfos.loc[_data.index, 'perc_visual_single'] = _data['perc_visual_single'].values * \
                _data['in_angle'].values
            master_dataframe_delfos.loc[_data.index, 'perc_visual_total'] = _data['perc_visual_total'].values
    
    with open('data/master_dataframe_delfos.pickle', 'wb') as handle:
        pickle.dump(master_dataframe_delfos, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# DELFOS PLOTS

#3D graph
g_list = [2021091300]
p_list = [902, 1075, 1174, 1226, 1250, 1293, 1344, 1368]

position = list(itertools.product(list(np.arange(1, 3)), list(np.arange(1, 5))))
cont = 0
fig = make_subplots(2, 4, vertical_spacing = 0.01, horizontal_spacing= 0.01)
for i in range(len(g_list)):
    _g = g_list[i]
    for j in range(len(p_list)):
        _p = p_list[j]
        _d_t = master_dataframe_delfos[master_dataframe_delfos['gameId'] == str(_g)].copy()
        _data = _d_t[_d_t['playId'] == str(_p)].copy()
        _data = _data.sort_values(by = 'distance').copy()
        _data = _data[_data['team'] != 'football'].copy()
        m_ = m_helmet * 3
        for i in range(0, len(_data)):
            if _data.loc[_data.index[i], 'in_angle'] != 0:
                xm_ym = np.array([_data.loc[_data.index[i], 'xmed'],
                                  _data.loc[_data.index[i], 'y_s'] + 
                                  0.5 * (_data.loc[_data.index[i], 'y_t'] -
                                  _data.loc[_data.index[i], 'y_s'])])
                perc = np.array([_data.loc[_data.index[i], 'perc_height'],
                                      _data.loc[_data.index[i], 'perc_width']])
                single_ = monigote(xm_ym, perc, height_, width_, mat_)
                if _data['team'].iloc[i] == _data['team_QB'].iloc[i]:
                    m_[(m_ == 0) & (single_ == 1)] = 1
                else:
                    m_[(m_ == 0) & (single_ == 1)] = 2
                # m_[m_!=0] = 1
        # m_ = m_ + m_helmet * 10
        fig.add_trace(
            go.Heatmap(z = m_, colorscale = ['white','red', '#009dff', 'black']), position[cont][0], position[cont][1])
        cont = cont + 1

for row_ in range(1,3):
    for col_ in range(1,5):
        fig.update_xaxes(showgrid = False, row = row_, col = col_)
        fig.update_yaxes(showgrid = False, row = row_, col = col_)
        fig.update_traces(showscale = False, row=row_, col = col_)
        fig.update_xaxes(zeroline = True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_yaxes(zeroline = True, zerolinewidth=1, zerolinecolor='#f2f4f4', mirror = True, row=row_, col=col_)
        fig.update_xaxes(showline = True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_yaxes(showline = True, linewidth=1, linecolor='black', row=row_, col=col_)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels = False, row=row_, col=col_)
        fig.update_yaxes(showticklabels = False, row=row_, col=col_)


fig.show()

game_id = g = 2021091300
play_id = p = 1250

# Filter data tables for plotting # 2D


_week_t = week_t[week_t['gameId'] == game_id].copy()
_week_t = _week_t[_week_t['playId'] == play_id].copy()
_week_t = _week_t[_week_t['event'] == 'pass_forward'].copy()

_plays = plays[plays['gameId'] == game_id].copy()
_plays = _plays[_plays['playId'] == play_id].copy()

_scout = scout[scout['gameId'] == game_id].copy()
_scout = _scout[_scout['playId'] == play_id].copy()

_offense = _plays['possessionTeam'].iloc[0]
_defense = _plays['defensiveTeam'].iloc[0]

_data = master_dataframe_delfos[master_dataframe_delfos['gameId'] == str(g)].copy()
_data = _data[_data['playId'] == str(p)].copy()

# Plot
figure = plotlines(_plays)
figure = plotfield(figure)
# figure = plotangle(figure, _week_t, _offense, _defense, players)
figure = plotplayersdl(figure, _week_t, _offense, _defense, players, _data)

figure.show()

#%% Line analytics

line_analytics = pd.DataFrame()

for g in np.unique(master_dataframe_delfos['gameId']):
    _d_delfos_g = master_dataframe_delfos[master_dataframe_delfos['gameId'] == str(g)].copy()
    for p in np.unique(_d_delfos_g['playId']):
        _d_delfos = _d_delfos_g[_d_delfos_g['playId'] == str(p)].copy()
        for i in range(len(_d_delfos)):
            if pd.isna(_d_delfos['closest_defender'].iloc[i]) == True:
                _d_delfos.loc[_d_delfos.index[i], 'closest_defender'] = _d_delfos['nflId'].iloc[i]
        
        line = np.sort(np.unique(_d_delfos['closest_defender']))
        line_st = [str(int(x)) for x in list(line)]
        line_st = '-'.join(list(line_st))
        
        line_analytics_tem = pd.DataFrame(index = np.arange(1), columns = ['gameId', 'playId', 'lineId',
                                                     'lineTeam', 'QB', 'perc_visual_total'])
         
        line_analytics_tem['gameId'] = _d_delfos['gameId'].iloc[0]
        line_analytics_tem['playId'] = _d_delfos['playId'].iloc[0]
        line_analytics_tem['lineId'] = line_st
        line_analytics_tem['lineTeam'] = ''
        line_analytics_tem['QB'] = _d_delfos['nflId_QB'].iloc[0]
        line_analytics_tem['perc_visual_total'] = _d_delfos['perc_visual_total'].iloc[0]
        
        line_analytics = pd.concat([line_analytics, line_analytics_tem])

line_analytics['DELFOS'] = 1 - line_analytics['perc_visual_total']
line_analytics['code'] = line_analytics['gameId'] + '-' + line_analytics['playId']

_plays = plays.copy()
_plays['gameId'] = _plays['gameId'].apply(lambda x: str(x))
_plays['playId'] = _plays['playId'].apply(lambda x: str(x))
_plays['code'] = _plays['gameId'] + '-' + _plays['playId']

line_analytics = line_analytics.merge(_plays[['code', 'passResult', 'possessionTeam']], left_on = 'code', right_on = 'code',
                                      how = 'left')
line_analytics = line_analytics[(line_analytics['passResult'] == 'C') | (line_analytics['passResult'] == 'I')].copy()
line_analytics.loc[line_analytics[line_analytics['passResult'] == 'C'].index, 'passResult'] = 1
line_analytics.loc[line_analytics[line_analytics['passResult'] == 'I'].index, 'passResult'] = 0
          


line_analytics_d = line_analytics.groupby('lineId').agg({'DELFOS':'mean', 'QB':'count', 'passResult': 'mean'})
line_analytics_d = line_analytics_d[line_analytics_d['QB'] >= 15].copy()
line_analytics_d_h = line_analytics_d.sort_values('DELFOS', ascending = False).head(10)
# line_analytics_d_h.reset_index(inplace = True) 
line_analytics_d_h.columns = ['DELFOS', 'NPlays', 'PassPerc']
line_analytics_d_h['DELFOS'] = line_analytics_d_h['DELFOS'].apply(lambda x: str(np.round(100 * x, 2)) + '%')
line_analytics_d_h['PassPerc'] = line_analytics_d_h['PassPerc'].apply(lambda x: str(np.round(100 * x, 2)) + '%')

fig = px.scatter(x = 100 * line_analytics_d['DELFOS'], y = 100 * line_analytics_d['passResult'], trendline="ols")
fig.update_yaxes(title_text="Pass accuracy (%)")
fig.update_xaxes(title_text="Visual Field Available")
# fig.update_yaxes(title_text="Pass accuracy (%)", secondary_y = True)
fig.update_xaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_yaxes(showline = True, linewidth=1, mirror = True, linecolor='black')
fig.update_layout(paper_bgcolor = 'white',
               plot_bgcolor = 'white')
fig.update_layout(xaxis_ticksuffix = '%', yaxis_tickformat = ',.')
fig.update_layout(yaxis_ticksuffix = '%',yaxis_tickformat = ',.')
# fig.update_layout(yaxis2_ticksuffix = '%', yaxis_tickformat = ',.')
# 
fig.update_traces(line_color = '#5a6e8b', line=dict(width=3))
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.update_layout( # customize font and legend orientation & position
    font_family = "Georgia",
    font_color="black",
    font_size = 15,
    legend=dict(
        font_size = 10, title=None, orientation = 'h', yanchor="bottom",
        y=1, x = 0.01, xanchor="left"))

# fig.update_layout(legend = dict(bgcolor = 'rgba(255,255,255,0.1)'))
fig.show()





line_analytics_t = line_analytics.groupby('possessionTeam').agg({'DELFOS':'mean'}).sort_values('DELFOS',ascending = False).head(10)
line_analytics_t['DELFOS'] = line_analytics_t['DELFOS'].apply(lambda x: str(np.round(100 * x, 2)) + '%')


