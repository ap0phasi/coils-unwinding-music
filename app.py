import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
from flask import Flask
import pygame
import numpy as np
import threading
from flask import Flask
import time
import plotly.graph_objs as go
from scipy.interpolate import interp1d

# coil setup
from coilspy import ComplexCoil

# Dash app setup
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)

num_tracks = 7

num_elements = num_tracks * 4 + 1

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Config", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
    ],
    brand="Ecological Services: Study on Unwinding",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 57,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "var(--sidebar-color)",
    "marginTop": '0px',
    "padding-top" : '10px',
    'color': 'white'
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 57,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "var(--secondary-bg-color)",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "margin-top" : '20px',
    "padding": "2rem 1rem",
    "background-color": "var(--secondary-bg-color)",
    "border-radius" : '10px',
    "box-shadow": "rgba(0, 0, 0, 0.25) 0px 54px 55px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px"
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "var(--secondary-bg-color)",
}


# Initialize Pygame mixer
pygame.mixer.init()

# Load the WAV files
orig_sounds = [pygame.mixer.Sound(f"files/{i+1}.wav") for i in range(num_tracks)]


def modify_sound(sound, pitch_factor, pan_factors):
    # load the sound into an array
    snd_array = pygame.sndarray.array(sound)

    # New length of the first dimension
    new_length = int(snd_array.shape[0] * pitch_factor)

    # Original indices
    original_indices = np.arange(snd_array.shape[0])

    # New indices for interpolation
    new_indices = np.linspace(0, snd_array.shape[0] - 1, new_length)

    # Interpolation
    resampled_array = np.zeros((new_length, snd_array.shape[1]), dtype=snd_array.dtype)
    for i in range(snd_array.shape[1]):
        interp_func = interp1d(original_indices, snd_array[:, i], kind='linear')
        resampled_array[:, i] = interp_func(new_indices) * pan_factors[i]

    snd_out = pygame.sndarray.make_sound(resampled_array)
    
    return snd_out

# Function to play music - placeholder for your actual implementation
def play_music(probs):
    new_sounds = orig_sounds
    sample_interval = int(probs[0]*10) #ms
    for i in range(num_tracks):
        selector_random = probs[i*4+1] * 10
        if selector_random > 0.2:
            pitch_factor = probs[i*4+2] * 40
            left_factor = probs[i*4+3] * 30
            right_factor = probs[i*4+4] * 30
            new_sound = modify_sound(new_sounds[i],pitch_factor = pitch_factor, pan_factors=[left_factor, right_factor])
            new_sound.play()
    time.sleep(sample_interval / 1000.0)  # Convert ms to seconds for sleep
        
# Function to start music in a separate thread to avoid blocking
def start_music(probs):
    music_thread = threading.Thread(target=play_music(probs), daemon=True)
    music_thread.start()
    
    
sidebar = html.Div(
    [
        html.H4("Configuration"),
        html.Hr(),
        html.P(
            "Configure coil parameters", className="lead"
        ),
        dbc.Button('Generate Coil',id='generate_coil_button', style = {'padding' : '10px'}),
        html.Div(id='coil-gen-output'),
        html.Div(style = {'padding':'20px'}),
        html.Label('State Angle'),
        dcc.Input(id='angles_state', type='number', value = 0.3, step=0.1),
        html.Label('Transition Angle'),
        dcc.Input(id='angles_transition', type='number', value = 0.2, step=0.1),
        html.Label('Interaction Angle'),
        dcc.Input(id='angles_interaction', type='number', value = 0.4, step=0.1),
        html.Label('State Center (Real)'),
        dcc.Input(id='state_center_re', type='number', value = 0, step=0.1),
        html.Label('State Center (Imaginary)'),
        dcc.Input(id='state_center_im', type='number', value = 0, step=0.1),
        html.Label('State Sigma (Real)'),
        dcc.Input(id='state_sigma_re', type='number', value = 1e100, step=0.1),
        html.Label('State Sigma (Imaginary)'),
        dcc.Input(id='state_sigma_im', type='number', value = 1e100, step=0.1),
        html.Label('State Magnitude (Real)'),
        dcc.Input(id='state_magnitude_re', type='number', value = 1.0, step=0.1),
        html.Label('State Magnitude (Imaginary)'),
        dcc.Input(id='state_magnitude_im', type='number', value = 100.0, step=0.1),
        
        html.Label('Transition Sigma Lower (Real)'),
        dcc.Input(id='trans_sigma_re_lo_lim', type='number', value = 0.2, step=0.1),
        html.Label('Transition Sigma Upper (Real)'),
        dcc.Input(id='trans_sigma_re_hi_lim', type='number', value = 0.5, step=0.1),
        html.Label('Transition Sigma (Imaginary)'),
        dcc.Input(id='trans_sigma_im', type='number', value = 1.0e10, step=0.1),
        html.Label('Transition Magnitude (Real)'),
        dcc.Input(id='trans_magnitude_re', type='number', value = 1.0, step=0.1),
        html.Label('Transition Magnitude (Imaginary)'),
        dcc.Input(id='trans_magnitude_im', type='number', value = 1.0e5, step=0.1),
        
        html.Label('Interaction Sigma Lower (Real)'),
        dcc.Input(id='inter_sigma_re_lo_lim', type='number', value = 0.2, step=0.1),
        html.Label('Interaction Sigma Upper (Real)'),
        dcc.Input(id='inter_sigma_re_hi_lim', type='number', value = 0.5, step=0.1),
        html.Label('Interaction Sigma (Imaginary)'),
        dcc.Input(id='inter_sigma_im', type='number', value = 1.0e10, step=0.1),
        html.Label('Interaction Magnitude (Real)'),
        dcc.Input(id='inter_magnitude_re', type='number', value = 1.0, step=0.1),
        html.Label('Interaction Magnitude (Imaginary)'),
        dcc.Input(id='inter_magnitude_im', type='number', value = 1.0e5, step=0.1),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(

    id="page-content",
    style=CONTENT_STYLE,
    children = [
            dcc.Store(id = 'coil-probs-store'),
            dbc.Button('Begin Analysis', id='play-button', n_clicks=0, style = {'margin': '10px', 'backgroundColor':'green', 'borderColor':'darkgreen'}),
            dbc.Button('Stop Analysis', id='stop-button', n_clicks=0, style = {'margin': '10px', 'backgroundColor':'red', 'borderColor':'darkred'}),
            dcc.Graph(id='live-update-graph'),
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ]
    )

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

@app.callback(
    Output('coil-gen-output','children'),
    Input('generate_coil_button','n_clicks'),
    State('angles_state','value'),
    State('angles_transition', 'value'),
    State('angles_interaction', 'value'),
    State('state_center_re','value'),
    State('state_center_im','value'),
    State('state_sigma_re', 'value'),
    State('state_sigma_im', 'value'),
    State('state_magnitude_re', 'value'),
    State('state_magnitude_im', 'value'),
    State('trans_sigma_re_lo_lim', 'value'),
    State('trans_sigma_re_hi_lim','value'),
    State('trans_sigma_im','value'),
    State('trans_magnitude_re', 'value'),
    State('trans_magnitude_im','value'),
    State('inter_sigma_re_lo_lim', 'value'),
    State('inter_sigma_re_hi_lim', 'value'),
    State('inter_sigma_im', 'value'),
    State('inter_magnitude_re', 'value'),
    State('inter_magnitude_im', 'value')
)
def generate_coil(n_clicks, 
                  angles_state,
                  angles_transition,
                  angles_interaction,
                  state_center_re,
                  state_center_im,
                  state_sigma_re,
                  state_sigma_im,
                  state_magnitude_re,
                  state_magnitude_im,
                  trans_sigma_re_lo_lim,
                  trans_sigma_re_hi_lim,
                  trans_sigma_im,
                  trans_mag_re,
                  trans_mag_im,
                  inter_sigma_re_lo_lim,
                  inter_sigma_re_hi_lim,
                  inter_sigma_im,
                  inter_mag_re,
                  inter_mag_im
                  ):
    if n_clicks:
        angles_dict = {
            'state': angles_state,
            'transition' : angles_transition,
            'interaction': angles_interaction
        }
        # No restrictions
        restrict_dict = {}

        global user_coil
        
        user_coil = ComplexCoil(
            num_elements=num_elements,
            angles_dict=angles_dict,
            state_center_re=state_center_re,
            state_center_im=state_center_im,
            state_sigma_re=state_sigma_re,
            state_sigma_im=state_sigma_im,
            state_magnitude_re=state_magnitude_re,
            state_magnitude_im=state_magnitude_im,
            state_restrictions=[],
            trans_sigma_re_lo_lim=trans_sigma_re_lo_lim,
            trans_sigma_re_hi_lim=trans_sigma_re_hi_lim,
            trans_sigma_im=trans_sigma_im,
            trans_mag_re=trans_mag_re,
            trans_mag_im=trans_mag_im,
            trans_restrictions=restrict_dict,
            inter_sigma_re_lo_lim=inter_sigma_re_lo_lim,
            inter_sigma_re_hi_lim=inter_sigma_re_hi_lim,
            inter_sigma_im=inter_sigma_im,
            inter_mag_re=inter_mag_re,
            inter_mag_im=inter_mag_im,
            inter_restrictions=restrict_dict,
        )
        
        return "Coil Generated"
    
# Callback to handle music control
@app.callback(
    Output('interval-component', 'disabled'),
    Input('play-button', 'n_clicks'),
    Input('stop-button', 'n_clicks')
)
def control_music(play_clicks, stop_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'play-button':
        return False
    elif button_id == 'stop-button':
        return True
    return True

# Callback to step through coils
@app.callback(Output('coil-probs-store', 'data'),
              Input('interval-component', 'n_intervals'),
              State('coil-probs-store', 'data'),
            )
def step_coil(n, data):
    
    if n > 0:
        data_orig = data
        # If there is no data stored yet then default
        data = data or np.empty((0, num_elements), dtype=float)
        # When the interval component triggers, we will step through the coil
        user_coil.step_coil(renormalize=True)
        probs = user_coil.get_prob().numpy()
        
        accumulated_data = np.vstack([data, probs])
        
        # Run the music
        if data_orig is not None:
            play_music(probs)

        return accumulated_data
    

# Callback to update the timeseries plot
@app.callback(Output('live-update-graph', 'figure'),
              Input('coil-probs-store', 'data'))
def update_graph_live(data):
    arr = np.vstack([data])

    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a trace for each column (series)
    for i in range(arr.shape[1]):
        fig.add_trace(go.Scatter(x=np.arange(arr.shape[0]),
                                y=arr[:, i],
                                mode='lines',
                                name=f'Series {i+1}'))

    # Update the layout
    fig.update_layout(title='Coil Dynamics',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    showlegend = False,
                    plot_bgcolor = 'rgb(129, 132, 130)',
                    paper_bgcolor = 'rgb(129, 132, 130)'
                    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, port=8086)