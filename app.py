import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Output, Input, State
from dash import callback_context, exceptions

# Load data
temp_data = pd.read_csv('temp-1901-2020-all.csv')
anomaly_data = pd.read_csv('HadCRUT4.csv')

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
x_vals = list(range(12))

country_options = temp_data['ISO3'].unique()
year_options = sorted(temp_data['Year'].unique())

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.Div([
        html.Label("Select Country:"),
        dcc.Dropdown(id='country-dropdown', options=[{'label': c, 'value': c} for c in country_options],
                     value='USA')
    ], style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Select Year:"),
        dcc.Slider(id='year-slider',
                   min=min(year_options),
                   max=max(year_options),
                   step=1,
                   value=2000,
                   marks={str(y): str(y) for y in range(min(year_options), max(year_options)+1, 10)},
                   tooltip={"placement": "bottom", "always_visible": True})
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '0px 20px 20px 20px'}),

    html.Button('Play', id='play-button', n_clicks=0, style={'margin': '10px'}),
    dcc.Interval(id='interval-component', interval=500, n_intervals=0, disabled=True),

    dcc.Graph(id='dashboard-graph')
])

# Callback to update graph
@app.callback(
    Output('dashboard-graph', 'figure'),
    Input('country-dropdown', 'value'),
    Input('year-slider', 'value')
)
def update_dashboard(country_iso, year):
    df = temp_data[(temp_data['ISO3'] == country_iso) & (temp_data['Year'] == year)]
    an = anomaly_data[anomaly_data['Year'] == year]
    temps = df['Temperature'].tolist()

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'polar'}, {'type': 'xy'}, {'type': 'polar'}]],
        column_widths=[0.3, 0.4, 0.3],
        subplot_titles=(
            f"{country_iso} {year} Monthly Temperature Distribution",
            f"{country_iso} {year} Monthly Temperature Variation",
            f"{year} World Temperature Anomaly"
        )
    )

    # Left: Barpolar
    fig.add_trace(go.Barpolar(
        theta=month_names,
        r=temps,
        base=[0]*12,
        marker=dict(
            color=temps,
            colorscale=[
                [0.0, "rgb(0,0,130)"],
                [0.25, "rgb(100,150,255)"],
                [0.5, "rgb(230,230,230)"],
                [0.75, "rgb(255,130,100)"],
                [1.0, "rgb(130,0,0)"]
            ],
            cmin=-40, cmax=30,
            line_color='white', line_width=2
        ),
        hovertemplate="%{theta}<br>%{r}°C<extra></extra>",
        opacity=0.8
    ), row=1, col=1)

    fig.update_polars(row=1, col=1,
                      bgcolor='rgba(0,0,0,0)',
                      radialaxis=dict(range=[-40, 30], tick0=-40, dtick=20,
                                      showline=True, ticks='outside', tickangle=0),
                      angularaxis=dict(direction='clockwise', rotation=90))

    # Middle: Line + Fill
    sign_flags = [t >= 0 for t in temps]
    runs, curr_sign, start = [], sign_flags[0], 0

    for i, flag in enumerate(sign_flags[1:], 1):
        if flag != curr_sign:
            runs.append((curr_sign, list(range(start, i))))
            curr_sign, start = flag, i
    runs.append((curr_sign, list(range(start, len(temps)))))

    for is_pos, idxs in runs:
        xs = [x_vals[i] for i in idxs]
        ys = [temps[i] for i in idxs]

        i0 = idxs[0]
        if i0 > 0 and temps[i0-1] * temps[i0] < 0:
            xL = (i0-1) + (-temps[i0-1]) / (temps[i0]-temps[i0-1])
        else:
            xL = xs[0]

        i1 = idxs[-1]
        if i1 < 11 and temps[i1] * temps[i1+1] < 0:
            xR = i1 + (-temps[i1]) / (temps[i1+1]-temps[i1])
        else:
            xR = xs[-1]

        poly_x = [xL] + xs + [xR] + [xR] + list(reversed(xs)) + [xL]
        poly_y = [0] + ys + [0] + [0] + [0]*len(xs) + [0]

        color = 'rgba(255,130,100,0.6)' if is_pos else 'rgba(65,105,225,0.4)'
        linec = 'black' if is_pos else 'royalblue'

        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color=linec, width=1),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_vals, y=temps,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hovertemplate="%{x}<br>%{y}°C<extra></extra>"
    ), row=1, col=2)

    fig.update_xaxes(row=1, col=2,
                     tickmode='array',
                     tickvals=x_vals,
                     ticktext=month_names,
                     tickangle=-90)

    # Right: Polar Anomaly
    if not an.empty:
        r_vals = an['Anomaly'].tolist()
        r_vals.append(r_vals[0])
        theta_vals = month_names + [month_names[0]]
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            mode='lines+markers',
            line_color='green',
            name='Anomaly',
            hovertemplate="%{theta}<br>%{r}°C<extra></extra>"
        ), row=1, col=3)
    else:
        fig.add_trace(go.Scatterpolar(
            r=[0]*12,
            theta=month_names,
            mode='lines',
            line_color='gray',
            name='No Anomaly Data'
        ), row=1, col=3)

    fig.update_polars(row=1, col=3,
                      bgcolor='rgba(0,0,0,0)',
                      radialaxis=dict(range=[-1, 1.2], tick0=-1, dtick=0.5,
                                      showline=True, ticks='outside', tickfont_size=12),
                      angularaxis=dict(direction='clockwise', rotation=90))

    fig.update_layout(
        showlegend=False,
        height=600,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)' 
    )

    return fig


@app.callback(
    Output('year-slider', 'value'),
    Output('interval-component', 'disabled'),
    Output('play-button',   'children'),
    Input('play-button',        'n_clicks'),
    Input('interval-component', 'n_intervals'),
    State('interval-component', 'disabled'),
    State('year-slider',          'value')
)
def control_playback(n_clicks, n_intervals, disabled, current_year):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    min_year = min(year_options)
    max_year = max(year_options)

    if trigger == 'play-button':
        if disabled:
            if current_year >= max_year:
                return min_year, False, 'Pause'
            return current_year, False, 'Pause'
        else:
            return current_year, True, 'Play'


    else:
        next_year = current_year + 1
        if next_year > max_year:
            return max_year, True, 'Play'
        return next_year, False, 'Pause'
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)