"""NiceGUI frontend"""
import os
from typing import Optional

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

    ui.run_with(
        fastapi_app,
        mount_path='/',  # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
        storage_secret=os.getenv("APP_SECRET"),
    )
