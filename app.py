from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import App_countries, App_marketing


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = 'Visual Data Analysis_Assignment'

# define a button style
button_style = {
    'font-size': '20px', 'text-align':'center',
    'width': '500px', 
    'height':'35px', 
    'display': 'inline-block',  
    'margin-left': '100px', 
    'background-color': 'darkgrey',
    'color': 'white'
    }

# create a layout 
app.layout = html.Div([ # create a container
    # represents the browser address bar and doesn't render anything
    dcc.Location(id='url', refresh=False), 
    html.H1( # layout's title
        children='Visual Data Analysis',
        style={'text-align':'center'}
        ),
    html.H2( # layout's subtitle
        children='Tatiana I., 2022-05-09',
        style={'text-align':'center'}
        ),
    dbc.Button( # create a button calling the 'Countries Data' layout
        'Death Causes and Socioeconomic Indicators', 
        href='/page1', 
        active='exact', 
        external_link=True,
        id='btn_1', 
        style=button_style
        ),
    dbc.Button( # create a button calling the 'Marketing Data' layout
        'Marketing Data', 
        href='/page2', 
        active='exact', 
        external_link=True,
        id='btn_2', 
        style=button_style
        ),
    html.Div(id='page-content') # render a content
])

# Update the index
@app.callback(
    Output('page-content', 'children'), 
    [Input('url', 'pathname')]
    )

def display_page(pathname: str):
    if pathname == '/page2':
        return App_marketing.layout
    else:
        return App_countries.layout

# host = '127.0.0.1'
if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1')

