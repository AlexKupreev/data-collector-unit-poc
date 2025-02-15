"""NiceGUI frontend"""
import os
from typing import Optional
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from nicegui.events import GenericEventArguments
from starlette.middleware.base import BaseHTTPMiddleware

from data_collector_unit_poc.web.scheduler import (
    build_now_trigger, 
    wrapped_ingest_noaa_isd_lite_job, 
    scheduler,
    get_running_jobs,
    terminate_job
)
from data_collector_unit_poc.jobs.weather import (
    get_noaa_isd_locations,
    build_local_noaa_isd_storage_path,
    read_data_parquet,
    read_data_orc_pandas,
)
from data_collector_unit_poc.settings import noaa_isd_local_persistent_path
import os


load_dotenv()

admin_user = os.getenv("ADMIN_USERNAME")
admin_pass = os.getenv("ADMIN_PASSWORD")

passwords = {admin_user: admin_pass}

unrestricted_page_routes = {'/login'}

def init(fastapi_app: FastAPI) -> None:
    
    class AuthMiddleware(BaseHTTPMiddleware):
        """This middleware restricts access to all NiceGUI pages.

        It redirects the user to the login page if they are not authenticated.
        """

        async def dispatch(self, request: Request, call_next):
            if not app.storage.user.get('authenticated', False):
                if not request.url.path.startswith('/_nicegui') and request.url.path not in unrestricted_page_routes:
                    app.storage.user['referrer_path'] = request.url.path  # remember where the user wanted to go
                    return RedirectResponse('/login')
            return await call_next(request)
        
    app.add_middleware(AuthMiddleware)
    
    @ui.page('/')
    def main_page() -> None:
        def logout() -> None:
            app.storage.user.clear()
            ui.navigate.to('/login')

        with ui.header(elevated=True).classes('items-center justify-between'):
            ui.label(f'Hello {app.storage.user["username"]}!').classes('text-2xl')
            with ui.row():
                ui.button('Jobs Management', on_click=lambda: ui.navigate.to('/jobs')).props('outline round')
                ui.button('Weather Locations', on_click=lambda: ui.navigate.to('/weather-locations')).props('outline round')
                ui.button(on_click=logout, icon='logout').props('outline round')

        ui.button('Ingest NOAA ISD', on_click=lambda: (
            scheduler.add_job(wrapped_ingest_noaa_isd_lite_job, build_now_trigger()), 
            ui.notify('Ingest NOAA ISD job triggered successfully!')
        )).props('outline round')

    @ui.page('/jobs')
    def jobs_page() -> None:
        def handle_terminate(job_id: str) -> None:
            if terminate_job(job_id):
                ui.notify('Job terminated successfully')
                refresh_jobs_table(jobs_container)
            else:
                ui.notify('Failed to terminate job', color='negative')

        with ui.header(elevated=True).classes('items-center justify-between'):
            ui.label('Jobs Management').classes('text-2xl')
            ui.button('Back to Home', on_click=lambda: ui.navigate.to('/')).props('outline round')

        # Create jobs grid
        jobs_container = ui.element('div').classes('w-full')

        def refresh_jobs_table(container):
            # Clear existing content
            container.clear()
            
            # Add header row
            with container:
                with ui.row().classes('w-full bg-blue-100 p-2'):
                    ui.label('Job ID').classes('flex-1')
                    ui.label('Name').classes('flex-1')
                    ui.label('Next Run Time').classes('flex-1')
                    ui.label('Trigger').classes('flex-1')
                    ui.label('Status').classes('flex-1')
                    ui.label('Actions').classes('flex-1')
            
            # Add job rows
            jobs = get_running_jobs()
            for job in jobs:
                with container:
                    with ui.row().classes('w-full p-2 border-b'):
                        ui.label(job['id']).classes('flex-1')
                        ui.label(job['name']).classes('flex-1')
                        ui.label(job['next_run_time'] if job['next_run_time'] else '-').classes('flex-1')
                        ui.label(job['trigger']).classes('flex-1')
                        ui.label(job['status']).classes('flex-1')
                        with ui.element('div').classes('flex-1'):
                            ui.button('Terminate', on_click=lambda _, job_id=job['id']: handle_terminate(job_id)) \
                                .props('outline round color=red')

        # Initial population
        refresh_jobs_table(jobs_container)

        # Add refresh button
        ui.button(
            'Refresh Jobs List', 
            on_click=lambda: refresh_jobs_table(jobs_container)
        ).props('outline round').classes('mt-4')

    @ui.page('/login')
    def login() -> Optional[RedirectResponse]:
        def try_login() -> None:  # local function to avoid passing username and password as arguments
            if passwords.get(username.value) == password.value:
                app.storage.user.update({'username': username.value, 'authenticated': True})
                ui.navigate.to(app.storage.user.get('referrer_path', '/'))  # go back to where the user wanted to go
            else:
                ui.notify('Wrong username or password', color='negative')

        if app.storage.user.get('authenticated', False):
            return RedirectResponse('/')
        with ui.card().classes('absolute-center'):
            username = ui.input('Username').on('keydown.enter', try_login)
            password = ui.input('Password', password=True, password_toggle_button=True).on('keydown.enter', try_login)
            ui.button('Log in', on_click=try_login)
        return None

    def read_location_data(location) -> pd.DataFrame | None:
        """Read weather data for a location from any available format"""
        base_filename = f"{location.usaf}-{location.wban}"
        
        # Try reading from different formats
        for format in ['parquet', 'orc', 'csv.gz']:
            filepath = os.path.join(noaa_isd_local_persistent_path, f"{base_filename}.{format}")
            if os.path.exists(filepath):
                try:
                    if format == 'parquet':
                        return read_data_parquet(filepath)
                    elif format == 'orc':
                        return read_data_orc_pandas(filepath)
                    else:  # csv.gz
                        return pd.read_csv(filepath, compression='gzip')
                except Exception as e:
                    print(f"Error reading {format} file: {e}")
                    continue
        return None

    def create_temperature_chart(df: pd.DataFrame, location_name: str):
        """Create a temperature timeline chart"""
        # Convert temperature to Celsius (data is stored in tenths of degrees Celsius)
        df['temperature'] = df['air_temp'] / 10.0
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # Calculate daily mean temperature
        daily_temp = df.groupby(df['datetime'].dt.date)['temperature'].mean().reset_index()
        
        # Create the chart
        fig = px.line(
            daily_temp, 
            x='datetime', 
            y='temperature',
            title=f'Mean Daily Temperature for {location_name}',
            labels={'temperature': 'Temperature (°C)', 'datetime': 'Date'}
        )
        return fig

    @ui.page('/weather-locations')
    def weather_locations_page() -> None:
        with ui.header(elevated=True).classes('items-center justify-between'):
            ui.label('Weather Locations').classes('text-2xl')
            ui.button('Back to Home', on_click=lambda: ui.navigate.to('/')).props('outline round')

        # Get locations and check which ones have data files
        locations = get_noaa_isd_locations()
        
        # Create locations grid
        locations_container = ui.element('div').classes('w-full')
        
        with locations_container:
            # Add header row
            with ui.row().classes('w-full bg-blue-100 p-2'):
                ui.label('Station').classes('flex-1')
                ui.label('ICAO').classes('flex-1')
                ui.label('Location').classes('flex-1')
                ui.label('Coordinates').classes('flex-1')
                ui.label('Elevation').classes('flex-1')
                ui.label('Period').classes('flex-1')
                ui.label('Data Available').classes('flex-1')
            
            # Add location rows
            for location in locations:
                # Check if any data file exists for this location
                base_filename = f"{location.usaf}-{location.wban}"
                has_data = any(
                    os.path.exists(os.path.join(noaa_isd_local_persistent_path, f"{base_filename}.{ext}"))
                    for ext in ['csv.gz', 'parquet', 'avro', 'orc']
                )
                
                with ui.row().classes('w-full p-2 border-b items-center'):
                    ui.label(location.station_name or '-').classes('flex-1')
                    ui.label(location.icao or '-').classes('flex-1')
                    ui.label(f"{location.country or '-'} {location.us_state or ''}".strip()).classes('flex-1')
                    ui.label(f"({location.lat}, {location.lon})").classes('flex-1')
                    ui.label(f"{location.elevation}m" if location.elevation else '-').classes('flex-1')
                    ui.label(f"{location.begin} - {location.end}").classes('flex-1')
                    with ui.element('div').classes('flex-1'):
                        if has_data:
                            ui.button('Show Chart', on_click=lambda loc=location: show_temperature_chart(loc)) \
                                .props('outline round color=blue')
                        else:
                            ui.label('✗')

        # Create a dialog for the chart
        chart_dialog = ui.dialog()
        chart_container = None

        def show_temperature_chart(location):
            nonlocal chart_container
            
            with chart_dialog:
                chart_dialog.clear()
                
                # Add header with close button
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label(f'Temperature Timeline: {location.station_name}').classes('text-xl')
                    ui.button('Close', on_click=chart_dialog.close).props('outline round')
                
                # Add loading indicator
                with ui.row().classes('w-full justify-center'):
                    ui.spinner('dots')
                    ui.label('Loading data...')
                
                # Show the dialog while loading
                chart_dialog.open()
                
                # Load and process the data
                df = read_location_data(location)
                if df is not None and not df.empty:
                    # Clear loading indicator
                    chart_dialog.clear()
                    
                    # Recreate header
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(f'Temperature Timeline: {location.station_name}').classes('text-xl')
                        ui.button('Close', on_click=chart_dialog.close).props('outline round')
                    
                    # Create and display the chart
                    fig = create_temperature_chart(df, location.station_name or str(location.usaf))
                    ui.plotly(fig).classes('w-full h-[500px]')
                else:
                    # Clear loading indicator and show error
                    chart_dialog.clear()
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label('Error').classes('text-xl text-red-500')
                        ui.button('Close', on_click=chart_dialog.close).props('outline round')
                    ui.label('Failed to load temperature data').classes('text-red-500')

    ui.run_with(
        fastapi_app,
        mount_path='/',  # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
        storage_secret=os.getenv("APP_SECRET"),
    )
